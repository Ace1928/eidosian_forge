from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from ..perturb.library import recipe_from_name, to_payloads
from ..types import TickContext, clamp01


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class ExperimentDesignerModule:
    name = "experiment_designer"

    def _should_run(self, ctx: TickContext, state: Mapping[str, Any]) -> bool:
        if not bool(ctx.config.get("experiment_designer_enabled", True)):
            return False
        interval = max(1, int(ctx.config.get("experiment_designer_interval_beats", 120)))
        last = int(state.get("last_emit_beat", -10_000))
        return (ctx.beat_count - last) >= interval

    def _safe_to_experiment(self, ctx: TickContext) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if ctx.perturbations_for("*"):
            reasons.append("active_perturbation")
        meta_evt = ctx.latest_event("meta.state_estimate")
        if isinstance(meta_evt, Mapping):
            data = meta_evt.get("data") if isinstance(meta_evt.get("data"), Mapping) else {}
            if str(data.get("mode") or "").lower() == "degraded":
                reasons.append("meta_degraded")
        err_count = len(ctx.latest_events("consciousness.module_error", k=32))
        if err_count > int(ctx.config.get("experiment_designer_max_recent_errors", 3)):
            reasons.append("module_error_spike")
        return (len(reasons) == 0), reasons

    def _select_recipe(self, deltas: Mapping[str, float]) -> tuple[str, str]:
        continuity = _safe_float(deltas.get("continuity_delta"), 0.0)
        ownership = _safe_float(deltas.get("ownership_delta"), 0.0)
        groundedness = _safe_float(deltas.get("groundedness_delta"), 0.0)
        trace_strength = _safe_float(deltas.get("trace_strength_delta"), 0.0)
        coherence = _safe_float(deltas.get("coherence_delta"), 0.0)
        prediction_error = _safe_float(deltas.get("prediction_error_delta"), 0.0)
        dream_likeness = _safe_float(deltas.get("dream_likeness_delta"), 0.0)

        if continuity < -0.04:
            return (
                "wm_lesion",
                "Hypothesis: working-set lesion explains continuity deficits.",
            )
        if ownership < -0.04:
            return (
                "identity_wobble",
                "Hypothesis: agency ownership instability drives report inconsistency.",
            )
        if groundedness < -0.04 and dream_likeness > 0.01:
            return (
                "sensory_deprivation",
                "Hypothesis: groundedness loss is tied to simulated-stream dominance.",
            )
        if trace_strength < -0.05:
            return (
                "attention_flood",
                "Hypothesis: noisy candidate competition is degrading ignition precision.",
            )
        if prediction_error > 0.04:
            return (
                "world_model_scramble",
                "Hypothesis: predictive coding instability is creating recurrent surprise spikes.",
            )
        if coherence < -0.04:
            return (
                "gw_bottleneck_strain",
                "Hypothesis: workspace bottleneck policy is over/under-constraining broadcast quality.",
            )
        return (
            "dopamine_spike",
            "Hypothesis: modulation dynamics are under-explored; exploration regime test requested.",
        )

    def tick(self, ctx: TickContext) -> None:
        state = ctx.module_state(
            self.name,
            defaults={
                "last_emit_beat": -10_000,
                "seen_trial_ids": [],
                "proposed_count": 0,
                "executed_count": 0,
                "last_recipe": "",
            },
        )
        if not self._should_run(ctx, state):
            return

        safe, reasons = self._safe_to_experiment(ctx)
        if not safe:
            state["last_emit_beat"] = int(ctx.beat_count)
            ctx.emit_event(
                "experiment.skipped",
                {"beat": int(ctx.beat_count), "reasons": reasons},
                tags=["consciousness", "experiment"],
            )
            return

        seen_raw = state.get("seen_trial_ids")
        seen = list(seen_raw) if isinstance(seen_raw, list) else []
        seen_set = {str(x) for x in seen}
        min_trials = max(1, int(ctx.config.get("experiment_designer_min_trials", 3)))
        recent = ctx.latest_events("bench.trial_result", k=64)
        unseen: list[Mapping[str, Any]] = []
        for evt in reversed(recent):
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            trial_id = str(data.get("trial_id") or "")
            if not trial_id or trial_id in seen_set:
                continue
            unseen.append(data)
            seen.append(trial_id)
            seen_set.add(trial_id)
            if len(unseen) >= min_trials:
                break

        if len(unseen) < min_trials:
            state["seen_trial_ids"] = seen[-300:]
            return

        agg: dict[str, float] = {}
        for row in unseen:
            deltas = row.get("deltas") if isinstance(row.get("deltas"), Mapping) else {}
            for key, value in deltas.items():
                agg[str(key)] = agg.get(str(key), 0.0) + _safe_float(value, 0.0)
        mean_deltas = {key: (val / float(len(unseen))) for key, val in agg.items()}

        recipe_name, hypothesis = self._select_recipe(mean_deltas)
        recipe = recipe_from_name(
            recipe_name,
            duration_s=float(ctx.config.get("experiment_designer_recipe_duration_s", 1.5)),
            magnitude=float(ctx.config.get("experiment_designer_recipe_magnitude", 0.35)),
        )
        if recipe is None:
            state["seen_trial_ids"] = seen[-300:]
            return

        payload = {
            "recipe": recipe.name,
            "description": recipe.description,
            "hypothesis": hypothesis,
            "mean_deltas": {k: round(v, 6) for k, v in mean_deltas.items()},
            "source_trial_ids": [str(r.get("trial_id") or "") for r in unseen],
        }
        proposal_evt = ctx.emit_event(
            "experiment.proposed",
            payload,
            tags=["consciousness", "experiment", "proposal"],
        )
        state["proposed_count"] = int(state.get("proposed_count") or 0) + 1
        state["last_recipe"] = recipe.name
        state["last_emit_beat"] = int(ctx.beat_count)
        state["seen_trial_ids"] = seen[-300:]

        if not bool(ctx.config.get("experiment_designer_auto_inject", False)):
            return

        corr_id = str(proposal_evt.get("corr_id") or "")
        parent_id = str(proposal_evt.get("corr_id") or "")
        injected_ids: list[str] = []
        for entry in to_payloads(recipe.perturbations):
            evt = ctx.emit_event(
                "perturb.inject",
                {
                    **entry,
                    "meta": {
                        **dict(entry.get("meta") or {}),
                        "source": "experiment_designer",
                        "recipe": recipe.name,
                    },
                },
                tags=["consciousness", "experiment", "perturb"],
                corr_id=corr_id or None,
                parent_id=parent_id or None,
            )
            injected_ids.append(str(evt.get("id") or ""))
        state["executed_count"] = int(state.get("executed_count") or 0) + 1
        ctx.emit_event(
            "experiment.executed",
            {
                "recipe": recipe.name,
                "injected_count": len(injected_ids),
                "injected_event_ids": injected_ids,
                "hypothesis": hypothesis,
                "confidence": clamp01(0.45 + (0.1 * len(unseen)), default=0.55),
            },
            tags=["consciousness", "experiment", "executed"],
            corr_id=corr_id or None,
            parent_id=parent_id or None,
        )


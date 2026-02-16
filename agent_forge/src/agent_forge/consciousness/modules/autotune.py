from __future__ import annotations

from typing import Any, Mapping

from ..tuning.optimizer import BanditOptimizer
from ..tuning.overlay import load_tuned_overlay, persist_tuned_overlay
from ..tuning.params import default_param_specs
from ..types import TickContext


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class AutotuneModule:
    name = "autotune"

    def _should_run(self, ctx: TickContext, state: Mapping[str, Any]) -> bool:
        if not bool(ctx.config.get("autotune_enabled", False)):
            return False
        interval = max(1, int(ctx.config.get("autotune_interval_beats", 120)))
        last_run = int(state.get("last_run_beat", -10_000))
        if (ctx.beat_count - last_run) < interval:
            return False
        return True

    def _safe_to_tune(self, ctx: TickContext) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if ctx.perturbations_for("*"):
            reasons.append("active_perturbations")

        meta = ctx.latest_event("meta.state_estimate")
        if isinstance(meta, Mapping):
            data = meta.get("data") if isinstance(meta.get("data"), Mapping) else {}
            if str(data.get("mode") or "").lower() == "degraded":
                reasons.append("meta_degraded")

        max_errors = int(ctx.config.get("autotune_guardrail_max_recent_errors", 2))
        errors = len(ctx.latest_events("consciousness.module_error", k=24))
        if errors > max_errors:
            reasons.append("module_error_spike")

        return (len(reasons) == 0), reasons

    def _run_micro_trial(
        self,
        ctx: TickContext,
        *,
        overlay: Mapping[str, Any],
        seed: int,
    ) -> Mapping[str, Any]:
        # Local import prevents kernel<->bench circular import at module load time.
        from ..bench.trials import ConsciousnessBenchRunner, TrialSpec

        runner = ConsciousnessBenchRunner(ctx.state_dir)
        spec = TrialSpec(
            name="autotune_micro",
            warmup_beats=int(ctx.config.get("autotune_trial_warmup_beats", 1)),
            baseline_seconds=float(ctx.config.get("autotune_trial_baseline_seconds", 1.5)),
            perturb_seconds=float(ctx.config.get("autotune_trial_perturb_seconds", 1.0)),
            recovery_seconds=float(ctx.config.get("autotune_trial_recovery_seconds", 1.5)),
            beat_seconds=float(ctx.config.get("autotune_trial_beat_seconds", 0.2)),
            task=str(ctx.config.get("autotune_task", "signal_pulse")),
            disable_modules=["autotune"],
            overlay=dict(overlay or {}),
            seed=max(0, int(seed)),
        )
        result = runner.run_trial(
            spec,
            persist=bool(ctx.config.get("autotune_persist_trials", True)),
        )
        return result.report

    def _guardrails(
        self, ctx: TickContext, report: Mapping[str, Any]
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if int(report.get("module_error_count") or 0) > int(
            ctx.config.get("autotune_guardrail_max_module_errors", 0)
        ):
            reasons.append("module_errors")

        if _safe_float(report.get("degraded_mode_ratio"), default=0.0) > _safe_float(
            ctx.config.get("autotune_guardrail_max_degraded_ratio"), default=0.45
        ):
            reasons.append("degraded_ratio")

        if int(report.get("winner_count") or 0) > int(
            ctx.config.get("autotune_guardrail_max_winner_count", 120)
        ):
            reasons.append("winner_flood")

        if int(report.get("ignitions_without_trace") or 0) > int(
            ctx.config.get("autotune_guardrail_max_trace_violations", 0)
        ):
            reasons.append("trace_violations")

        return (len(reasons) == 0), reasons

    def tick(self, ctx: TickContext) -> None:
        state = ctx.module_state(
            self.name,
            defaults={
                "last_run_beat": -10_000,
                "baseline_score": None,
                "best_score": None,
                "best_overlay": {},
                "optimizer_state": {},
                "trial_count": 0,
                "accepted_count": 0,
            },
        )
        if not self._should_run(ctx, state):
            return

        safe, reasons = self._safe_to_tune(ctx)
        if not safe:
            state["last_run_beat"] = int(ctx.beat_count)
            ctx.emit_event(
                "tune.skipped",
                {
                    "beat": int(ctx.beat_count),
                    "reasons": reasons,
                },
                tags=["consciousness", "autotune"],
            )
            return

        specs = default_param_specs()
        current_overlay, invalid_keys = load_tuned_overlay(ctx.state_store)
        if invalid_keys:
            ctx.emit_event(
                "consciousness.param_invalid",
                {"invalid_keys": invalid_keys},
                tags=["consciousness", "autotune", "config"],
            )
        optimizer = BanditOptimizer(
            param_specs=specs,
            state=state["optimizer_state"],  # type: ignore[arg-type]
            rng=ctx.rng,
        )

        base_seed = int(ctx.config.get("autotune_seed_offset", 1_000_000))
        baseline_score = state.get("baseline_score")
        if baseline_score is None:
            baseline_report = self._run_micro_trial(
                ctx,
                overlay=current_overlay,
                seed=base_seed + int(ctx.beat_count),
            )
            baseline_score = _safe_float(baseline_report.get("composite_score"), default=0.0)
            state["baseline_score"] = baseline_score
            state["best_score"] = baseline_score
            state["best_overlay"] = current_overlay
            ctx.emit_event(
                "tune.baseline",
                {
                    "score": baseline_score,
                    "trial_id": baseline_report.get("trial_id"),
                },
                tags=["consciousness", "autotune"],
            )

        proposal = optimizer.propose(current_overlay)
        proposal_overlay = proposal.get("overlay") if isinstance(proposal, Mapping) else {}
        if not isinstance(proposal_overlay, Mapping):
            proposal_overlay = {}
        ctx.emit_event(
            "tune.proposed",
            {
                "beat": int(ctx.beat_count),
                "key": proposal.get("key"),
                "before": proposal.get("before"),
                "after": proposal.get("after"),
            },
            tags=["consciousness", "autotune"],
        )

        trial_report = self._run_micro_trial(
            ctx,
            overlay=proposal_overlay,
            seed=base_seed + int(ctx.beat_count) + 1,
        )
        score = _safe_float(trial_report.get("composite_score"), default=0.0)
        guard_ok, guard_reasons = self._guardrails(ctx, trial_report)

        best_score = _safe_float(state.get("best_score"), default=_safe_float(baseline_score, default=0.0))
        min_improvement = _safe_float(ctx.config.get("autotune_min_improvement"), default=0.03)
        accepted = bool(guard_ok and score >= (best_score + min_improvement))
        optimizer.observe(accepted=accepted)

        state["trial_count"] = int(state.get("trial_count") or 0) + 1
        state["last_run_beat"] = int(ctx.beat_count)
        state["last_score"] = score
        state["last_trial_id"] = str(trial_report.get("trial_id") or "")
        baseline = _safe_float(state.get("baseline_score"), default=0.0)
        state["baseline_score"] = round((0.9 * baseline) + (0.1 * score), 6)

        if accepted:
            persisted = persist_tuned_overlay(
                ctx.state_store,
                proposal_overlay,
                source="autotune",
                reason=f"score_improved_by_{round(score - best_score, 6)}",
                score=score,
            )
            state["best_score"] = score
            state["best_overlay"] = dict(persisted.get("overlay") or {})
            state["accepted_count"] = int(state.get("accepted_count") or 0) + 1
            ctx.emit_event(
                "tune.commit",
                {
                    "score": score,
                    "best_score": score,
                    "version": persisted.get("version"),
                    "key": proposal.get("key"),
                    "before": proposal.get("before"),
                    "after": proposal.get("after"),
                    "trial_id": trial_report.get("trial_id"),
                },
                tags=["consciousness", "autotune", "commit"],
            )
        else:
            ctx.emit_event(
                "tune.rollback",
                {
                    "score": score,
                    "best_score": best_score,
                    "reasons": list(guard_reasons) if guard_reasons else ["no_improvement"],
                    "key": proposal.get("key"),
                    "before": proposal.get("before"),
                    "after": proposal.get("after"),
                    "trial_id": trial_report.get("trial_id"),
                },
                tags=["consciousness", "autotune", "rollback"],
            )

        ctx.metric(
            "consciousness.autotune.best_score",
            _safe_float(state.get("best_score"), default=0.0),
        )
        ctx.metric("consciousness.autotune.last_score", score)
        trials = int(state.get("trial_count") or 0)
        accepts = int(state.get("accepted_count") or 0)
        ratio = float(accepts) / float(trials) if trials > 0 else 0.0
        ctx.metric("consciousness.autotune.acceptance_ratio", ratio)


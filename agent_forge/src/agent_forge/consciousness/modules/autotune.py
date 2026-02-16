from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Protocol

from ..tuning.bayes_optimizer import BayesParetoOptimizer
from ..tuning.objectives import objectives_from_trial_report
from ..tuning.optimizer import BanditOptimizer
from ..tuning.overlay import load_tuned_overlay, persist_tuned_overlay
from ..tuning.params import default_param_specs
from ..types import TickContext


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class _Optimizer(Protocol):
    def propose(self, current_overlay: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def observe_result(
        self,
        *,
        accepted: bool,
        score: float | None = None,
        overlay: Mapping[str, Any] | None = None,
        objectives: Mapping[str, float] | None = None,
        report: Mapping[str, Any] | None = None,
    ) -> None: ...


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

    def _build_optimizer(
        self,
        *,
        mode: str,
        specs: Mapping[str, Any],
        state: MutableMapping[str, Any],
        rng: Any,
        config: Mapping[str, Any],
    ) -> _Optimizer:
        key = str(mode or "bandit").strip().lower().replace("-", "_")
        state["optimizer_kind"] = key
        if key in {"bayes", "bayes_pareto", "bayesian"}:
            state["candidate_pool"] = int(
                config.get("autotune_bayes_candidate_pool", state.get("candidate_pool", 14))
            )
            state["kernel_gamma"] = float(
                config.get("autotune_bayes_kernel_gamma", state.get("kernel_gamma", 3.5))
            )
            state["kappa"] = float(
                config.get("autotune_bayes_kappa", state.get("kappa", 0.35))
            )
            state["exploration"] = float(
                config.get("autotune_bayes_exploration", state.get("exploration", 0.12))
            )
            return BayesParetoOptimizer(param_specs=specs, state=state, rng=rng)
        state["step_scale"] = float(
            config.get("autotune_bandit_step_scale", state.get("step_scale", 0.2))
        )
        return BanditOptimizer(param_specs=specs, state=state, rng=rng)

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

    def _run_red_team_guard(
        self,
        ctx: TickContext,
        *,
        overlay: Mapping[str, Any],
        seed: int,
    ) -> tuple[bool, list[str], dict[str, Any]]:
        enabled = bool(ctx.config.get("autotune_run_red_team", True))
        if not enabled:
            return True, [], {"enabled": False, "available": True}

        quick = bool(ctx.config.get("autotune_red_team_quick", True))
        max_scenarios = max(0, int(ctx.config.get("autotune_red_team_max_scenarios", 1)))
        persist = bool(ctx.config.get("autotune_red_team_persist", False))
        min_pass_ratio = _safe_float(
            ctx.config.get("autotune_red_team_min_pass_ratio"), default=0.75
        )
        min_robustness = _safe_float(
            ctx.config.get("autotune_red_team_min_robustness"), default=0.70
        )
        require_available = bool(
            ctx.config.get("autotune_red_team_require_available", True)
        )
        configured_disable = [
            str(name)
            for name in list(ctx.config.get("autotune_red_team_disable_modules") or [])
            if str(name)
        ]
        merged_disable = sorted({self.name, *configured_disable})

        report: dict[str, Any]
        reasons: list[str] = []
        try:
            from ..bench.red_team import ConsciousnessRedTeamCampaign

            campaign = ConsciousnessRedTeamCampaign(ctx.state_dir)
            campaign_report = campaign.run(
                persist=persist,
                base_seed=max(0, int(seed)),
                max_scenarios=max_scenarios,
                quick=quick,
                overlay=dict(overlay or {}),
                disable_modules=merged_disable,
            ).report
            report = dict(campaign_report)
            report["enabled"] = True
            report["available"] = True
        except Exception as exc:
            report = {
                "enabled": True,
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
                "pass_ratio": 0.0,
                "mean_robustness": 0.0,
            }
            if require_available:
                reasons.append("red_team_unavailable")
            ctx.emit_event(
                "tune.red_team_error",
                {
                    "error": report["error"],
                    "quick": quick,
                    "max_scenarios": max_scenarios,
                    "require_available": require_available,
                },
                tags=["consciousness", "autotune", "red_team", "error"],
            )

        pass_ratio = _safe_float(report.get("pass_ratio"), default=0.0)
        robustness = _safe_float(report.get("mean_robustness"), default=0.0)
        if report.get("available", False):
            if pass_ratio < min_pass_ratio:
                reasons.append("red_team_pass_ratio")
            if robustness < min_robustness:
                reasons.append("red_team_robustness")

        guard_ok = len(reasons) == 0
        guard_payload = {
            "guard_ok": guard_ok,
            "reasons": reasons,
            "pass_ratio": pass_ratio,
            "mean_robustness": robustness,
            "min_pass_ratio": min_pass_ratio,
            "min_robustness": min_robustness,
            "quick": quick,
            "max_scenarios": max_scenarios,
            "seed": int(seed),
            "require_available": require_available,
            "available": bool(report.get("available", False)),
            "overlay_keys": sorted(str(k) for k in dict(overlay or {}).keys()),
            "disable_modules": merged_disable,
            "run_id": str(report.get("run_id") or ""),
            "scenario_count": int(report.get("scenario_count") or 0),
        }
        ctx.emit_event(
            "tune.red_team",
            guard_payload,
            tags=["consciousness", "autotune", "red_team"],
        )
        return guard_ok, reasons, report

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
        optimizer_mode = str(ctx.config.get("autotune_optimizer", "bayes_pareto"))
        optimizer = self._build_optimizer(
            mode=optimizer_mode,
            specs=specs,
            state=state["optimizer_state"],  # type: ignore[arg-type]
            rng=ctx.rng,
            config=ctx.config,
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
                "optimizer": optimizer_mode,
                "acquisition": proposal.get("acquisition"),
            },
            tags=["consciousness", "autotune"],
        )

        trial_report = self._run_micro_trial(
            ctx,
            overlay=proposal_overlay,
            seed=base_seed + int(ctx.beat_count) + 1,
        )
        score = _safe_float(trial_report.get("composite_score"), default=0.0)
        objectives = objectives_from_trial_report(trial_report)
        guard_ok, guard_reasons = self._guardrails(ctx, trial_report)
        red_team_seed = int(ctx.config.get("autotune_red_team_seed_offset", 2_000_000)) + int(
            ctx.beat_count
        )

        best_score = _safe_float(state.get("best_score"), default=_safe_float(baseline_score, default=0.0))
        min_improvement = _safe_float(ctx.config.get("autotune_min_improvement"), default=0.03)
        improved = score >= (best_score + min_improvement)
        red_team_ok = True
        red_team_reasons: list[str] = []
        red_team_report: dict[str, Any] = {"enabled": False, "available": True}
        if guard_ok and improved:
            red_team_ok, red_team_reasons, red_team_report = self._run_red_team_guard(
                ctx,
                overlay=proposal_overlay,
                seed=red_team_seed,
            )
        elif bool(ctx.config.get("autotune_run_red_team", True)):
            red_team_reasons.append("red_team_skipped")

        accepted = bool(guard_ok and improved and red_team_ok)
        optimizer.observe_result(
            accepted=accepted,
            score=score,
            overlay=proposal_overlay,
            objectives=objectives,
            report=trial_report,
        )

        state["trial_count"] = int(state.get("trial_count") or 0) + 1
        state["last_run_beat"] = int(ctx.beat_count)
        state["last_score"] = score
        state["last_trial_id"] = str(trial_report.get("trial_id") or "")
        baseline = _safe_float(state.get("baseline_score"), default=0.0)
        state["baseline_score"] = round((0.9 * baseline) + (0.1 * score), 6)
        state["last_red_team"] = {
            "enabled": bool(red_team_report.get("enabled", False)),
            "available": bool(red_team_report.get("available", False)),
            "pass_ratio": _safe_float(red_team_report.get("pass_ratio"), default=0.0),
            "mean_robustness": _safe_float(red_team_report.get("mean_robustness"), default=0.0),
            "run_id": str(red_team_report.get("run_id") or ""),
            "reasons": list(red_team_reasons),
        }

        if accepted:
            persisted = persist_tuned_overlay(
                ctx.state_store,
                proposal_overlay,
                source="autotune",
                reason=f"{optimizer_mode}_score_improved_by_{round(score - best_score, 6)}",
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
                    "optimizer": optimizer_mode,
                    "objectives": objectives,
                    "red_team": {
                        "enabled": bool(red_team_report.get("enabled", False)),
                        "available": bool(red_team_report.get("available", False)),
                        "run_id": str(red_team_report.get("run_id") or ""),
                        "pass_ratio": _safe_float(red_team_report.get("pass_ratio"), default=0.0),
                        "mean_robustness": _safe_float(
                            red_team_report.get("mean_robustness"), default=0.0
                        ),
                    },
                },
                tags=["consciousness", "autotune", "commit"],
            )
        else:
            rollback_reasons: list[str] = []
            rollback_reasons.extend(str(r) for r in guard_reasons if str(r))
            if not improved:
                rollback_reasons.append("no_improvement")
            rollback_reasons.extend(str(r) for r in red_team_reasons if str(r))
            if not rollback_reasons:
                rollback_reasons.append("no_improvement")
            deduped_reasons: list[str] = []
            seen: set[str] = set()
            for reason in rollback_reasons:
                if reason in seen:
                    continue
                seen.add(reason)
                deduped_reasons.append(reason)
            ctx.emit_event(
                "tune.rollback",
                {
                    "score": score,
                    "best_score": best_score,
                    "reasons": deduped_reasons,
                    "key": proposal.get("key"),
                    "before": proposal.get("before"),
                    "after": proposal.get("after"),
                    "trial_id": trial_report.get("trial_id"),
                    "optimizer": optimizer_mode,
                    "objectives": objectives,
                    "red_team": {
                        "enabled": bool(red_team_report.get("enabled", False)),
                        "available": bool(red_team_report.get("available", False)),
                        "run_id": str(red_team_report.get("run_id") or ""),
                        "pass_ratio": _safe_float(red_team_report.get("pass_ratio"), default=0.0),
                        "mean_robustness": _safe_float(
                            red_team_report.get("mean_robustness"), default=0.0
                        ),
                    },
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
        if bool(red_team_report.get("enabled", False)):
            ctx.metric(
                "consciousness.autotune.red_team_pass_ratio",
                _safe_float(red_team_report.get("pass_ratio"), default=0.0),
            )
            ctx.metric(
                "consciousness.autotune.red_team_robustness",
                _safe_float(red_team_report.get("mean_robustness"), default=0.0),
            )

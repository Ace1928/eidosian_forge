from __future__ import annotations

from typing import Any, Mapping, Optional

from agent_forge.core import workspace

from ..types import TickContext, WorkspacePayload, normalize_workspace_payload


def _event_data(evt: Mapping[str, Any]) -> Mapping[str, Any]:
    return evt.get("data") if isinstance(evt.get("data"), Mapping) else {}


def _latest_metric(events: list[Mapping[str, Any]], key: str) -> Optional[float]:
    for evt in reversed(events):
        if str(evt.get("type") or "") != "metrics.sample":
            continue
        data = _event_data(evt)
        if str(data.get("key") or "") != key:
            continue
        try:
            return float(data.get("value"))
        except (TypeError, ValueError):
            return None
    return None


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / max(len(values), 1)


class MetaModule:
    name = "meta"

    def __init__(self) -> None:
        self._last_mode: str | None = None
        self._last_confidence: float | None = None

    def _mode_from_signals(
        self,
        *,
        coherence: float,
        mean_prediction_error: float,
        report_groundedness: float,
        ignition_burst: float,
        source_gini: float,
        simulated_fraction: float,
    ) -> tuple[str, float, list[str]]:
        disconfirmers: list[str] = []
        if coherence < 0.2:
            disconfirmers.append("low_coherence")
        if mean_prediction_error > 0.65:
            disconfirmers.append("high_prediction_error")
        if report_groundedness < 0.4:
            disconfirmers.append("low_report_groundedness")
        if ignition_burst < 1.0:
            disconfirmers.append("weak_ignition_burst")
        if source_gini > 0.75:
            disconfirmers.append("source_dominance")
        if simulated_fraction > 0.5:
            disconfirmers.append("high_simulated_fraction")

        grounded_cond = (
            coherence >= 0.2
            and mean_prediction_error <= 0.5
            and report_groundedness >= 0.5
            and source_gini <= 0.75
            and simulated_fraction <= 0.5
        )
        if grounded_cond:
            conf = (0.45 + 0.42 * coherence) + (0.08 * min(1.0, ignition_burst / 2.0))
            return "grounded", max(0.0, min(1.0, conf)), disconfirmers
        if simulated_fraction > 0.5 and mean_prediction_error <= 0.8:
            conf = (0.42 + 0.25 * min(1.0, simulated_fraction)) + (
                0.15 * (1.0 - min(1.0, mean_prediction_error))
            )
            return "simulated", max(0.0, min(1.0, conf)), disconfirmers
        if coherence >= 0.1 and mean_prediction_error <= 0.75:
            conf = (0.4 + 0.35 * coherence) + (0.08 * min(1.0, ignition_burst / 2.0))
            return "simulated", max(0.0, min(1.0, conf)), disconfirmers
        confidence = max(
            0.0, min(1.0, 0.35 + (1.0 - min(1.0, mean_prediction_error)) * 0.3)
        )
        return "degraded", confidence, disconfirmers

    def tick(self, ctx: TickContext) -> None:
        lookback = int(ctx.config.get("meta_observation_window", 160))
        emit_delta = float(ctx.config.get("meta_emit_delta_threshold", 0.05))

        events = list(ctx.recent_events)[-lookback:] + list(ctx.emitted_events)
        ws = workspace.summary(
            ctx.state_dir, limit=350, window_seconds=1.0, min_sources=3
        )
        coherence = float(ws.get("coherence_ratio") or 0.0)
        ignition_burst = float(ws.get("max_ignition_burst") or 0.0)
        source_gini = float(ws.get("source_gini") or 0.0)
        report_groundedness = _latest_metric(
            events, "consciousness.report.groundedness"
        )
        if report_groundedness is None:
            report_groundedness = 0.0

        pred_errors: list[float] = []
        sim_count = 0
        real_percept_count = 0
        for evt in events:
            etype = str(evt.get("type") or "")
            if etype == "world.prediction_error":
                data = _event_data(evt)
                try:
                    pred_errors.append(float(data.get("prediction_error")))
                except (TypeError, ValueError):
                    pass
            elif etype == "sense.simulated_percept":
                sim_count += 1
            elif etype == "sense.percept":
                real_percept_count += 1
        mean_prediction_error = _mean(pred_errors[-40:])
        total_percepts = sim_count + real_percept_count
        simulated_fraction = (sim_count / total_percepts) if total_percepts > 0 else 0.0

        mode, confidence, disconfirmers = self._mode_from_signals(
            coherence=coherence,
            mean_prediction_error=mean_prediction_error,
            report_groundedness=report_groundedness,
            ignition_burst=ignition_burst,
            source_gini=source_gini,
            simulated_fraction=simulated_fraction,
        )
        confidence = round(float(confidence), 6)

        should_emit = False
        if self._last_mode is None or self._last_confidence is None:
            should_emit = True
        elif self._last_mode != mode:
            should_emit = True
        elif abs(confidence - self._last_confidence) >= emit_delta:
            should_emit = True

        self._last_mode = mode
        self._last_confidence = confidence

        mode_code = {"degraded": 0.0, "simulated": 0.5, "grounded": 1.0}.get(mode, 0.0)
        ctx.metric("consciousness.meta.mode_code", mode_code)
        ctx.metric("consciousness.meta.confidence", confidence)

        if not should_emit:
            return

        evt = ctx.emit_event(
            "meta.state_estimate",
            {
                "mode": mode,
                "confidence": confidence,
                "signals": {
                    "coherence_ratio": round(coherence, 6),
                    "mean_prediction_error": round(mean_prediction_error, 6),
                    "report_groundedness": round(report_groundedness, 6),
                    "ignition_count": int(ws.get("ignition_count") or 0),
                    "max_ignition_burst": round(ignition_burst, 6),
                    "source_gini": round(source_gini, 6),
                    "simulated_fraction": round(simulated_fraction, 6),
                    "simulated_percept_count": int(sim_count),
                    "real_percept_count": int(real_percept_count),
                },
                "disconfirmers": disconfirmers,
            },
            tags=["consciousness", "meta", "state"],
        )

        payload = WorkspacePayload(
            kind="META",
            source_module="meta",
            content={
                "mode": mode,
                "confidence": confidence,
                "signals": {
                    "coherence_ratio": round(coherence, 6),
                    "mean_prediction_error": round(mean_prediction_error, 6),
                    "max_ignition_burst": round(ignition_burst, 6),
                    "source_gini": round(source_gini, 6),
                    "simulated_fraction": round(simulated_fraction, 6),
                },
                "disconfirmers": disconfirmers,
            },
            confidence=confidence,
            salience=max(0.2, 1.0 - min(1.0, mean_prediction_error)),
            links={
                "corr_id": evt.get("corr_id"),
                "parent_id": evt.get("parent_id"),
                "memory_ids": [],
            },
        ).as_dict()
        payload = normalize_workspace_payload(
            payload, fallback_kind="META", source_module="meta"
        )
        ctx.broadcast(
            "meta",
            payload,
            tags=["consciousness", "meta", "broadcast"],
            corr_id=evt.get("corr_id"),
            parent_id=evt.get("parent_id"),
        )

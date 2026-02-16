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
    ) -> tuple[str, float, list[str]]:
        disconfirmers: list[str] = []
        if coherence < 0.2:
            disconfirmers.append("low_coherence")
        if mean_prediction_error > 0.65:
            disconfirmers.append("high_prediction_error")
        if report_groundedness < 0.4:
            disconfirmers.append("low_report_groundedness")

        if coherence >= 0.2 and mean_prediction_error <= 0.5 and report_groundedness >= 0.5:
            return "grounded", max(0.0, min(1.0, 0.5 + 0.5 * coherence)), disconfirmers
        if coherence >= 0.1 and mean_prediction_error <= 0.75:
            return "simulated", max(0.0, min(1.0, 0.45 + 0.4 * coherence)), disconfirmers
        confidence = max(0.0, min(1.0, 0.35 + (1.0 - min(1.0, mean_prediction_error)) * 0.3))
        return "degraded", confidence, disconfirmers

    def tick(self, ctx: TickContext) -> None:
        lookback = int(ctx.config.get("meta_observation_window", 160))
        emit_delta = float(ctx.config.get("meta_emit_delta_threshold", 0.05))

        events = list(ctx.recent_events)[-lookback:] + list(ctx.emitted_events)
        ws = workspace.summary(ctx.state_dir, limit=350, window_seconds=1.0, min_sources=3)
        coherence = float(ws.get("coherence_ratio") or 0.0)
        report_groundedness = _latest_metric(events, "consciousness.report.groundedness")
        if report_groundedness is None:
            report_groundedness = 0.0

        pred_errors: list[float] = []
        for evt in events:
            if str(evt.get("type") or "") != "world.prediction_error":
                continue
            data = _event_data(evt)
            try:
                pred_errors.append(float(data.get("prediction_error")))
            except (TypeError, ValueError):
                continue
        mean_prediction_error = _mean(pred_errors[-40:])

        mode, confidence, disconfirmers = self._mode_from_signals(
            coherence=coherence,
            mean_prediction_error=mean_prediction_error,
            report_groundedness=report_groundedness,
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
        payload = normalize_workspace_payload(payload, fallback_kind="META", source_module="meta")
        ctx.broadcast(
            "meta",
            payload,
            tags=["consciousness", "meta", "broadcast"],
            corr_id=evt.get("corr_id"),
            parent_id=evt.get("parent_id"),
        )

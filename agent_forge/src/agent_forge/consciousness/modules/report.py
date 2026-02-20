from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Mapping, Optional

from ..types import TickContext, WorkspacePayload, normalize_workspace_payload


def _event_data(evt: Mapping[str, Any]) -> Mapping[str, Any]:
    return evt.get("data") if isinstance(evt.get("data"), Mapping) else {}


def _winner_from_event(evt: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    data = _event_data(evt)
    payload = data.get("payload") if isinstance(data.get("payload"), Mapping) else {}
    if str(payload.get("kind") or "") != "GW_WINNER":
        return None
    content = payload.get("content") if isinstance(payload.get("content"), Mapping) else {}
    links = payload.get("links") if isinstance(payload.get("links"), Mapping) else {}
    return {
        "candidate_id": content.get("candidate_id"),
        "winner_candidate_id": content.get("winner_candidate_id") or links.get("winner_candidate_id"),
        "source_event_type": content.get("source_event_type"),
        "source_module": content.get("source_module"),
        "score": content.get("score"),
        "salience": payload.get("salience"),
        "confidence": payload.get("confidence"),
        "corr_id": links.get("corr_id"),
        "parent_id": links.get("parent_id"),
    }


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / max(len(values), 1)


class ReportModule:
    name = "report"

    def __init__(self) -> None:
        self._last_groundedness: Optional[float] = None
        self._last_emit_at: Optional[datetime] = None

    def tick(self, ctx: TickContext) -> None:
        emit_interval = float(ctx.config.get("report_emit_interval_secs", 2.0))
        emit_delta = float(ctx.config.get("report_emit_delta_threshold", 0.08))
        min_broadcast_groundedness = float(ctx.config.get("report_broadcast_min_groundedness", 0.35))
        index = ctx.index
        winner = None
        winner_broadcasts = index.broadcasts_by_kind.get("GW_WINNER") or []
        if winner_broadcasts:
            winner = _winner_from_event(winner_broadcasts[-1])
        action_evt = index.latest_by_type.get("policy.action")
        agency_evt = index.latest_by_type.get("self.agency_estimate")
        boundary_evt = index.latest_by_type.get("self.boundary_estimate")
        meta_evt = index.latest_by_type.get("meta.state_estimate")
        pred_err_evt = index.latest_by_type.get("world.prediction_error")
        simulated_evt = index.latest_by_type.get("sense.simulated_percept")

        if not action_evt and not winner:
            return

        action_data = _event_data(action_evt) if action_evt else {}
        agency_data = _event_data(agency_evt) if agency_evt else {}
        boundary_data = _event_data(boundary_evt) if boundary_evt else {}
        meta_data = _event_data(meta_evt) if meta_evt else {}
        pred_err_data = _event_data(pred_err_evt) if pred_err_evt else {}
        simulated_data = _event_data(simulated_evt) if simulated_evt else {}

        winner_candidate = str((winner or {}).get("candidate_id") or "")
        action_candidate = str(action_data.get("selected_candidate_id") or "")
        candidate_match = 1.0 if winner_candidate and winner_candidate == action_candidate else 0.0

        agency_conf = _clamp01(agency_data.get("agency_confidence"), default=0.0)
        boundary_stability = _clamp01(boundary_data.get("boundary_stability"), default=0.0)
        prediction_error = _clamp01(pred_err_data.get("prediction_error"), default=0.5)
        meta_confidence = _clamp01(meta_data.get("confidence"), default=0.0)
        meta_mode = str(meta_data.get("mode") or "unknown")
        simulation_active = bool(simulated_data) and (
            bool(simulated_data.get("simulated"))
            or str(simulated_data.get("origin") or "").startswith("world_model.rollout")
        )
        groundedness = round(
            _mean(
                [
                    candidate_match,
                    agency_conf,
                    boundary_stability,
                    1.0 - prediction_error,
                    meta_confidence,
                ]
            ),
            6,
        )
        if meta_mode == "simulated" and simulation_active:
            groundedness = round(max(0.0, groundedness * 0.85), 6)

        disconfirmers: list[str] = []
        if candidate_match < 1.0:
            disconfirmers.append("winner_action_mismatch")
        if agency_conf < 0.5:
            disconfirmers.append("low_agency_confidence")
        if boundary_stability < 0.5:
            disconfirmers.append("low_boundary_stability")
        if prediction_error > 0.7:
            disconfirmers.append("high_prediction_error")
        if meta_confidence < 0.4:
            disconfirmers.append("low_meta_confidence")
        if simulation_active:
            disconfirmers.append("simulated_context_active")

        should_emit = False
        if self._last_emit_at is None:
            should_emit = True
        else:
            elapsed = (ctx.now - self._last_emit_at).total_seconds()
            if elapsed >= emit_interval:
                should_emit = True
        if self._last_groundedness is None:
            should_emit = True
        elif abs(groundedness - self._last_groundedness) >= emit_delta:
            should_emit = True
        if not should_emit:
            return

        report_id = uuid.uuid4().hex
        content = {
            "report_id": report_id,
            "mode": meta_mode,
            "groundedness": groundedness,
            "summary": {
                "action_kind": action_data.get("action_kind"),
                "selected_candidate_id": action_candidate,
                "winner_candidate_id": winner_candidate,
                "prediction_error": prediction_error,
                "simulation_active": simulation_active,
                "simulated_percept_origin": simulated_data.get("origin") if simulation_active else None,
                "simulated_percept_type": simulated_data.get("predicted_event_type") if simulation_active else None,
            },
            "confidence_breakdown": {
                "candidate_match": candidate_match,
                "agency_confidence": agency_conf,
                "boundary_stability": boundary_stability,
                "meta_confidence": meta_confidence,
            },
            "disconfirmers": disconfirmers,
            "evidence": {
                "action_corr_id": action_evt.get("corr_id") if action_evt else None,
                "agency_corr_id": agency_evt.get("corr_id") if agency_evt else None,
                "meta_corr_id": meta_evt.get("corr_id") if meta_evt else None,
                "prediction_corr_id": pred_err_evt.get("corr_id") if pred_err_evt else None,
                "simulation_corr_id": simulated_evt.get("corr_id") if simulated_evt else None,
            },
        }
        corr_id = str(action_evt.get("corr_id") or "") if action_evt else ""
        parent_id = str(action_evt.get("parent_id") or "") if action_evt else ""

        evt = ctx.emit_event(
            "report.self_report",
            content,
            tags=["consciousness", "report", "self_report"],
            corr_id=corr_id or None,
            parent_id=parent_id or None,
        )
        ctx.metric("consciousness.report.groundedness", groundedness)

        if groundedness >= min_broadcast_groundedness:
            payload = WorkspacePayload(
                kind="REPORT",
                source_module="report",
                content=content,
                confidence=groundedness,
                salience=max(0.25, groundedness),
                links={
                    "corr_id": evt.get("corr_id"),
                    "parent_id": evt.get("parent_id"),
                    "memory_ids": [],
                },
            ).as_dict()
            payload = normalize_workspace_payload(payload, fallback_kind="REPORT", source_module="report")
            ctx.broadcast(
                "report",
                payload,
                tags=["consciousness", "report", "broadcast"],
                corr_id=evt.get("corr_id"),
                parent_id=evt.get("parent_id"),
            )

        self._last_emit_at = ctx.now
        self._last_groundedness = groundedness

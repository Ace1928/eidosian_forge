from __future__ import annotations

import uuid
from typing import Any, Dict, Mapping, Optional

from ..types import TickContext, WorkspacePayload, normalize_workspace_payload


def _winner_from_broadcast(evt: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if evt.get("type") != "workspace.broadcast":
        return None
    data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
    payload = data.get("payload") if isinstance(data.get("payload"), Mapping) else {}
    if payload.get("kind") != "GW_WINNER":
        return None
    content = payload.get("content") if isinstance(payload.get("content"), Mapping) else {}
    return {
        "candidate_id": content.get("candidate_id"),
        "source_event_type": content.get("source_event_type"),
        "source_module": content.get("source_module"),
        "score": content.get("score"),
        "salience": payload.get("salience"),
        "confidence": payload.get("confidence"),
    }


def _select_action_kind(source_event_type: str) -> str:
    if source_event_type.startswith("world.prediction_error"):
        return "reconcile_prediction"
    if source_event_type.startswith("intero.drive"):
        return "regulate_drive"
    if source_event_type.startswith("sense."):
        return "inspect_signal"
    if source_event_type.startswith("report."):
        return "verify_report"
    return "observe"


class PolicyModule:
    name = "policy"

    def _latest_winner(self, ctx: TickContext) -> Optional[Dict[str, Any]]:
        for evt in reversed(ctx.emitted_events):
            winner = _winner_from_broadcast(evt)
            if winner:
                return winner
        for evt in reversed(ctx.recent_broadcasts):
            winner = _winner_from_broadcast(evt)
            if winner:
                return winner
        return None

    def tick(self, ctx: TickContext) -> None:
        max_actions = int(ctx.config.get("policy_max_actions_per_tick", 1))
        emit_broadcast = bool(ctx.config.get("policy_emit_broadcast", True))

        winner = self._latest_winner(ctx)
        if not winner:
            return

        action_count = 0
        while action_count < max_actions:
            action_count += 1
            action_id = uuid.uuid4().hex
            source_event_type = str(winner.get("source_event_type") or "")
            action_kind = _select_action_kind(source_event_type)
            confidence = float(winner.get("confidence") or 0.5)
            salience = float(winner.get("salience") or 0.5)

            predicted_observation = {
                "expected_event_type": "workspace.broadcast",
                "expected_source": "workspace_competition",
                "expected_kind": "GW_WINNER",
                "expected_candidate_id": str(winner.get("candidate_id") or ""),
            }

            action_evt = ctx.emit_event(
                "policy.action",
                {
                    "action_id": action_id,
                    "action_kind": action_kind,
                    "selected_candidate_id": winner.get("candidate_id"),
                    "selected_source_event_type": source_event_type,
                    "selected_source_module": winner.get("source_module"),
                    "score": winner.get("score"),
                    "confidence": confidence,
                    "salience": salience,
                },
                tags=["consciousness", "policy"],
                corr_id=action_id,
                parent_id=action_id,
            )

            ctx.emit_event(
                "policy.efference",
                {
                    "action_id": action_id,
                    "predicted_observation": predicted_observation,
                    "confidence": confidence,
                },
                tags=["consciousness", "policy", "efference"],
                corr_id=action_evt.get("corr_id"),
                parent_id=action_evt.get("parent_id"),
            )

            if emit_broadcast:
                payload = WorkspacePayload(
                    kind="PLAN",
                    source_module="policy",
                    content={
                        "action_id": action_id,
                        "action_kind": action_kind,
                        "candidate_id": winner.get("candidate_id"),
                        "source_event_type": source_event_type,
                    },
                    confidence=confidence,
                    salience=salience,
                    links={
                        "corr_id": action_evt.get("corr_id"),
                        "parent_id": action_evt.get("parent_id"),
                        "memory_ids": [],
                    },
                ).as_dict()
                payload = normalize_workspace_payload(payload, fallback_kind="PLAN", source_module="policy")
                ctx.broadcast(
                    "policy",
                    payload,
                    tags=["consciousness", "policy", "plan"],
                    corr_id=action_evt.get("corr_id"),
                    parent_id=action_evt.get("parent_id"),
                )

            ctx.metric("consciousness.policy.actions", 1.0)
            return

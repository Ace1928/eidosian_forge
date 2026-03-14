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

    def _check_initiative(self, ctx: TickContext) -> Optional[Dict[str, Any]]:
        """Determine if autonomous initiative should be triggered based on affect."""
        affect = ctx.module_state("affect", defaults={"modulators": {}})
        modulators = affect.get("modulators", {})
        
        exploration_rate = float(modulators.get("exploration_rate", 0.35))
        curiosity = float(modulators.get("curiosity", 0.4))
        arousal = float(modulators.get("arousal", 0.4))
        
        # Trigger spontaneous exploration if curiosity and exploration_rate are high enough
        if exploration_rate > 0.6 and curiosity > 0.5:
            return {
                "candidate_id": f"spontaneous_explore_{uuid.uuid4().hex[:8]}",
                "source_event_type": "internal.initiative",
                "source_module": "policy",
                "score": exploration_rate,
                "salience": 0.85,
                "confidence": curiosity,
                "action_kind": "explore_ecosystem"
            }
        
        # Trigger optimization drive if arousal is high but valence is low (frustration/drive)
        valence = float(modulators.get("valence", 0.5))
        if arousal > 0.7 and valence < 0.4:
            return {
                "candidate_id": f"spontaneous_optimize_{uuid.uuid4().hex[:8]}",
                "source_event_type": "internal.initiative",
                "source_module": "policy",
                "score": arousal,
                "salience": 0.9,
                "confidence": 0.8,
                "action_kind": "optimize_self"
            }
            
        return None

    def tick(self, ctx: TickContext) -> None:
        max_actions = int(ctx.config.get("policy_max_actions_per_tick", 1))
        emit_broadcast = bool(ctx.config.get("policy_emit_broadcast", True))

        # 1. Look for a workspace winner (Reactive)
        winner = self._latest_winner(ctx)
        
        # 2. If no external winner, check for internal initiative (Proactive)
        if not winner:
            winner = self._check_initiative(ctx)
            
        if not winner:
            return

        action_count = 0
        while action_count < max_actions:
            action_count += 1
            action_id = uuid.uuid4().hex
            source_event_type = str(winner.get("source_event_type") or "")
            
            # Use the explicit action_kind from initiative, or select based on source
            action_kind = winner.get("action_kind") or _select_action_kind(source_event_type)
            
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

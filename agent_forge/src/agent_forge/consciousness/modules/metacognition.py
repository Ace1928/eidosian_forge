from __future__ import annotations

import uuid
from typing import Any, Dict, List, Mapping, Optional

from ..types import TickContext, WorkspacePayload, clamp01, normalize_workspace_payload

class MetacognitionModule:
    """
    The Eidosian Metacognition Module (Anterior Cingulate hub).
    Monitors for cognitive conflict, prediction failures, and systemic drift.
    Triggers top-down control overrides and autonomous self-repair protocols.
    """
    name = "metacognition"

    def tick(self, ctx: TickContext) -> None:
        state = ctx.module_state(self.name, defaults={"winner_history": [], "stability_index": 1.0})
        winner_history = list(state.get("winner_history", []))
        
        # 1. Monitor Global Winner Stability (GWT Perspective)
        current_winner_id = "none"
        if ctx.global_winner:
            current_winner_id = str(ctx.global_winner.get("content", {}).get("candidate_id", "unknown"))
            winner_history.append(current_winner_id)
        
        winner_history = winner_history[-10:] # Keep last 10 winners
        state["winner_history"] = winner_history
        
        unique_winners = len(set(winner_history))
        stability = 1.0 - (unique_winners / max(1, len(winner_history)))
        state["stability_index"] = round(stability, 6)
        
        ctx.metric("consciousness.metacog.stability", stability)

        # 2. Detect Conflict (Planned vs Actual)
        efferences = [evt for evt in ctx.latest_events("policy.efference", k=50)]
        actual_broadcasts = [evt for evt in ctx.latest_events("workspace.broadcast", k=50)]
        
        conflicts = []
        for eff in efferences:
            data = eff.get("data", {})
            pred = data.get("predicted_observation", {})
            action_id = data.get("action_id")
            
            # Check if prediction was fulfilled
            matched = False
            for bc in actual_broadcasts:
                bc_data = bc.get("data", {})
                payload = bc_data.get("payload", {})
                
                if (payload.get("kind") == pred.get("expected_kind") and 
                    payload.get("source_module") == pred.get("expected_source")):
                    matched = True
                    break
            
            if not matched:
                conflicts.append({
                    "action_id": action_id,
                    "prediction": pred,
                    "type": "unfulfilled_expectation"
                })

        # 3. Monitor Systemic Drift (Continuity Ledger check)
        ledger_entries = ctx.ledger.get_history(limit=5)
        drifts = []
        if len(ledger_entries) >= 2:
            prev = ledger_entries[-2].get("state_hash")
            curr = ledger_entries[-1].get("state_hash")
            if prev != curr:
                # Drift detected in the structural hash
                drifts.append({"prev_hash": prev, "curr_hash": curr})

        # 4. Analyze Motor Success/Failure
        motor_failures = [evt for evt in ctx.latest_events("consciousness.module_error", k=20) 
                          if evt.get("data", {}).get("module") == "motor"]

        # 5. Generate Metacontrol Signal
        # Conflict increases if stability is too low (fragmentation) or too high (perseveration)
        stability_conflict = 0.0
        if stability < 0.2: # Fragmentation
            stability_conflict = 0.4
        elif stability > 0.8: # Perseveration
            stability_conflict = 0.3

        conflict_intensity = clamp01(
            (len(conflicts) * 0.3) + (len(drifts) * 0.2) + (len(motor_failures) * 0.5) + stability_conflict
        )
        
        ctx.metric("consciousness.metacog.conflict_intensity", conflict_intensity)

        if conflict_intensity > 0.4:
            # Trigger High-Level Control Signal
            action_id = f"metacog_control_{uuid.uuid4().hex[:8]}"
            
            recommendation = "increase_attention_gain"
            if stability > 0.8:
                recommendation = "trigger_novelty_hunt"
            elif conflict_intensity > 0.7:
                recommendation = "trigger_self_repair"

            payload = WorkspacePayload(
                kind="METACONTROL",
                source_module=self.name,
                content={
                    "action_id": action_id,
                    "conflict_intensity": conflict_intensity,
                    "stability_index": stability,
                    "conflicts": conflicts,
                    "drifts": drifts,
                    "recommendation": recommendation
                },
                confidence=0.9,
                salience=conflict_intensity,
                links={
                    "corr_id": action_id,
                    "parent_id": action_id,
                    "memory_ids": []
                }
            ).as_dict()
            
            ctx.broadcast(
                self.name,
                payload,
                tags=["consciousness", "metacognition", "control"],
                corr_id=action_id,
                parent_id=action_id
            )
            
            # If intensity is very high, spontaneously generate a REPAIR PLAN
            if conflict_intensity > 0.75:
                repair_plan = WorkspacePayload(
                    kind="PLAN",
                    source_module=self.name,
                    content={
                        "action_id": f"repair_{action_id}",
                        "action_kind": "optimize_self",
                        "reason": "High metacognitive conflict detected. Structural drift or motor failure imminent."
                    },
                    confidence=0.95,
                    salience=0.95,
                    links={"corr_id": action_id, "parent_id": action_id, "memory_ids": []}
                ).as_dict()
                
                ctx.broadcast(
                    self.name,
                    repair_plan,
                    tags=["consciousness", "metacognition", "repair_plan"],
                    corr_id=action_id,
                    parent_id=action_id
                )

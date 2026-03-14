from __future__ import annotations

from typing import Any, Dict, List

from ..types import TickContext, WorkspacePayload

try:
    import os
    from pathlib import Path

    from agent_forge.autonomy.gates import SystemicGateKeeper

    _forge_dir = Path(os.environ.get("EIDOS_FORGE_DIR", str(Path(__file__).resolve().parents[5]))).resolve()
    _gk = SystemicGateKeeper(repo_root=_forge_dir, invariants_path=_forge_dir / "GEMINI.md")
except ImportError:
    _gk = None


class MotorModule:
    """
    The Executive/Motor module of the Eidosian Consciousness.
    Translates 'PLAN' payloads from the Policy module into actual systemic actions.
    """

    name = "motor"

    def tick(self, ctx: TickContext) -> None:
        # 1. Listen for 'PLAN' payloads in the recent broadcasts
        plans = self._collect_plans(ctx)
        if not plans:
            return

        for plan in plans:
            # 2. Translate the plan into an execution intent
            action_kind = plan.get("action_kind")
            action_id = plan.get("action_id")
            # 3. Autonomous Execution: Handle specific autonomous actions
            if action_kind == "optimize_self" and _gk is not None:
                proposal_id = _gk.propose_change(
                    target_path="agent_forge/src/agent_forge/consciousness/config.json",  # Heuristic target
                    change_type="config",
                    proposed_content='{"autonomy_enabled": true}',
                    rationale="Spontaneous autonomous optimization triggered by high arousal and drive.",
                )
                action_kind = f"optimize_self_proposed_{proposal_id}"
            elif action_kind == "consolidate_memory":
                # DMN triggered memory consolidation (Hippocampal Replay)
                # In a live system, this sends an event for the memory daemon to pick up.
                action_kind = "consolidate_memory_initiated"

            # 4. Emit a 'motor.execution' event to signal intent to the system
            ctx.emit_event(
                "motor.execution",
                {"action_id": action_id, "action_kind": action_kind, "status": "initiated", "source_plan": plan},
                tags=["consciousness", "motor", "execution"],
                corr_id=action_id,
                parent_id=action_id,
            )

            # 4. Broadcast the intent so the phenomenology_probe can observe it
            payload = WorkspacePayload(
                kind="MOTOR_INTENT",
                source_module=self.name,
                content={"action_id": action_id, "action_kind": action_kind, "status": "active"},
                confidence=0.9,
                salience=0.8,
                links={"corr_id": action_id, "parent_id": action_id, "memory_ids": []},
            ).as_dict()

            ctx.broadcast(
                self.name, payload, tags=["consciousness", "motor", "broadcast"], corr_id=action_id, parent_id=action_id
            )

            # 5. Absolute Traceability: Log autonomous motor intent to the Continuity Ledger
            ctx.ledger.record_heartbeat(
                {
                    "motor_execution": True,
                    "action_id": action_id,
                    "action_kind": action_kind,
                    "source_plan": plan,
                    "beat_count": ctx.beat_count,
                }
            )

            ctx.metric("consciousness.motor.executions", 1.0)

    def _collect_plans(self, ctx: TickContext) -> List[Dict[str, Any]]:
        plans = []
        # Check both emitted events and recent broadcasts
        for evt in list(ctx.emitted_events) + list(ctx.recent_broadcasts):
            if evt.get("type") == "workspace.broadcast":
                data = evt.get("data", {})
                payload = data.get("payload", {})
                if payload.get("kind") == "PLAN" and payload.get("source_module") in ("policy", "default_mode"):
                    plans.append(payload.get("content", {}))
        return plans

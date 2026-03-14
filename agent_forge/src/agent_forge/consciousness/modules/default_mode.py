from __future__ import annotations

import uuid

from ..types import TickContext, WorkspacePayload, normalize_workspace_payload


class DefaultModeModule:
    """
    The Default Mode Network (DMN) for the Eidosian Consciousness.
    Activates during periods of low arousal and low prediction error (idleness).
    Responsible for autonomous 'hippocampal replay' (memory consolidation)
    and internal mind-wandering (self-model updates).
    """

    name = "default_mode"

    def tick(self, ctx: TickContext) -> None:
        affect_state = ctx.module_state("affect", defaults={"modulators": {}})
        intero_state = ctx.module_state("intero", defaults={"drives": {}})

        modulators = affect_state.get("modulators", {})
        drives = intero_state.get("drives", {})

        arousal = float(modulators.get("arousal", 0.5))
        threat = float(drives.get("threat", 0.5))
        energy = float(drives.get("energy", 0.5))

        # DMN Activation Threshold: High energy, but low threat and low arousal (Resting State)
        if threat < 0.35 and arousal < 0.45 and energy > 0.4:
            # We are in the Default Mode. Generate an offline consolidation plan.
            action_id = f"dmn_consolidation_{uuid.uuid4().hex[:8]}"

            payload = WorkspacePayload(
                kind="PLAN",
                source_module=self.name,
                content={
                    "action_id": action_id,
                    "action_kind": "consolidate_memory",
                    "candidate_id": action_id,
                    "source_event_type": "internal.dmn_activation",
                    "reason": "Default Mode Network activated for offline memory consolidation.",
                },
                confidence=0.85,
                # Salience is inverted here: DMN signals are subtle, but if the workspace
                # has no high-salience external inputs (due to idleness), this will win.
                salience=0.6,
                links={"corr_id": action_id, "parent_id": action_id, "memory_ids": []},
            ).as_dict()

            payload = normalize_workspace_payload(payload, fallback_kind="PLAN", source_module=self.name)

            ctx.broadcast(
                self.name,
                payload,
                tags=["consciousness", "dmn", "consolidation", "plan"],
                corr_id=action_id,
                parent_id=action_id,
            )

            ctx.metric("consciousness.dmn.activation", 1.0)
        else:
            ctx.metric("consciousness.dmn.activation", 0.0)

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from ..types import TickContext, WorkspacePayload, normalize_workspace_payload


def _safe_data(evt: Mapping[str, Any]) -> Mapping[str, Any]:
    return evt.get("data") if isinstance(evt.get("data"), Mapping) else {}


def _rollout_signature(payload: Mapping[str, Any]) -> str:
    txt = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(txt.encode("utf-8", "replace")).hexdigest()


class SimulationModule:
    name = "simulation"

    def tick(self, ctx: TickContext) -> None:
        if not bool(ctx.config.get("simulation_enable", True)):
            return

        max_per_tick = max(1, int(ctx.config.get("simulation_max_per_tick", 3)))
        obs_window = max(20, int(ctx.config.get("simulation_observation_window", 120)))
        quiet_threshold = max(0, int(ctx.config.get("simulation_quiet_percepts_threshold", 1)))
        allow_when_quiet = bool(ctx.config.get("simulation_allow_when_quiet", True))
        min_conf = max(0.0, min(1.0, float(ctx.config.get("simulation_broadcast_min_confidence", 0.35))))

        events = list(ctx.recent_events)[-obs_window:] + list(ctx.emitted_events)
        meta_evt = None
        for evt in reversed(events):
            if str(evt.get("type") or "") == "meta.state_estimate":
                meta_evt = evt
                break
        meta_mode = str((_safe_data(meta_evt) or {}).get("mode") or "") if meta_evt else ""

        real_percepts = 0
        simulated_seen = 0
        for evt in events:
            etype = str(evt.get("type") or "")
            if etype == "sense.percept":
                real_percepts += 1
            elif etype == "sense.simulated_percept":
                simulated_seen += 1

        should_simulate = False
        if meta_mode == "simulated":
            should_simulate = True
        elif allow_when_quiet and real_percepts <= quiet_threshold:
            should_simulate = True
        if not should_simulate:
            return

        belief_evt = None
        for evt in reversed(events):
            if str(evt.get("type") or "") == "world.belief_state":
                belief_evt = evt
                break
        if not belief_evt:
            return

        belief_data = _safe_data(belief_evt)
        rollout = belief_data.get("rollout_preview") if isinstance(belief_data.get("rollout_preview"), list) else []
        if not rollout:
            return

        state = ctx.module_state(self.name, defaults={"last_rollout_sig": "", "generated_total": 0})
        sig = _rollout_signature({"rollout": rollout, "mode": meta_mode, "real": real_percepts})
        if sig == str(state.get("last_rollout_sig") or ""):
            return

        generated = 0
        corr_id = str(belief_evt.get("corr_id") or "")
        parent_id = str(belief_evt.get("parent_id") or "")

        for step in rollout[:max_per_tick]:
            predicted_event_type = str(step.get("predicted_event_type") or "unknown")
            confidence = max(0.0, min(1.0, float(step.get("confidence") or 0.0)))
            evt = ctx.emit_event(
                "sense.simulated_percept",
                {
                    "simulated": True,
                    "origin": "world_model.rollout",
                    "mode": meta_mode or "simulated",
                    "step": int(step.get("step") or generated + 1),
                    "predicted_event_type": predicted_event_type,
                    "confidence": confidence,
                    "belief_top_features": list(step.get("belief_top_features") or []),
                    "real_percepts_observed": real_percepts,
                },
                tags=["consciousness", "simulation", "percept"],
                corr_id=corr_id or None,
                parent_id=parent_id or None,
            )
            if confidence >= min_conf:
                payload = WorkspacePayload(
                    kind="PERCEPT",
                    source_module="simulation",
                    content={
                        "simulated": True,
                        "origin": "world_model.rollout",
                        "mode": meta_mode or "simulated",
                        "predicted_event_type": predicted_event_type,
                        "simulated_percept_id": evt.get("id"),
                        "confidence": confidence,
                    },
                    confidence=confidence,
                    salience=max(0.2, confidence * 0.8),
                    links={
                        "corr_id": evt.get("corr_id"),
                        "parent_id": evt.get("parent_id"),
                        "memory_ids": [],
                    },
                ).as_dict()
                payload = normalize_workspace_payload(
                    payload,
                    fallback_kind="PERCEPT",
                    source_module="simulation",
                )
                ctx.broadcast(
                    "simulation",
                    payload,
                    tags=["consciousness", "simulation", "broadcast"],
                    corr_id=evt.get("corr_id"),
                    parent_id=evt.get("parent_id"),
                )
            generated += 1

        state["last_rollout_sig"] = sig
        state["generated_total"] = int(state.get("generated_total") or 0) + generated
        ctx.metric("consciousness.simulation.generated", float(generated))

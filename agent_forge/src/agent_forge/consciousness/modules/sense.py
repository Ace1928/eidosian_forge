from __future__ import annotations

from typing import Any, Mapping

from ..types import TickContext
from ..types import WorkspacePayload, clamp01, normalize_workspace_payload


class SenseModule:
    name = "sense"

    def _event_signature(self, evt: Mapping[str, Any]) -> str:
        return f"{evt.get('ts','')}|{evt.get('type','')}|{evt.get('corr_id','')}"

    def _should_perceive(self, etype: str) -> bool:
        if not etype:
            return False
        if etype.startswith(
            (
                "sense.",
                "attn.",
                "gw.",
                "policy.",
                "self.",
                "meta.",
                "report.",
                "wm.",
                "metrics.",
                "mem.",
                "knowledge.",
                "memory_bridge.",
                "knowledge_bridge.",
            )
        ):
            return False
        return True

    def tick(self, ctx: TickContext) -> None:
        max_percepts = max(1, int(ctx.config.get("sense_max_percepts_per_tick", 6)))
        emit_threshold = clamp01(ctx.config.get("sense_emit_threshold"), default=0.72)
        scan_events = max(20, int(ctx.config.get("sense_scan_events", 220)))
        perturbations = ctx.perturbations_for(self.name)
        if any(str(p.get("kind") or "") == "drop" for p in perturbations):
            return
        if any(str(p.get("kind") or "") == "delay" for p in perturbations) and (ctx.beat_count % 2 == 1):
            return
        noise_mag = max(
            [
                clamp01(p.get("magnitude"), default=0.0)
                for p in perturbations
                if str(p.get("kind") or "") == "noise"
            ]
            or [0.0]
        )
        clamp_floor = max(
            [
                clamp01(p.get("magnitude"), default=0.0)
                for p in perturbations
                if str(p.get("kind") or "") == "clamp"
            ]
            or [0.0]
        )
        scramble = any(str(p.get("kind") or "") == "scramble" for p in perturbations)

        state = ctx.module_state(
            self.name,
            defaults={
                "seen_signatures": [],
                "event_type_counts": {},
            },
        )
        seen = set(str(x) for x in list(state.get("seen_signatures") or []))
        counts = (
            state.get("event_type_counts")
            if isinstance(state.get("event_type_counts"), Mapping)
            else {}
        )
        counts_map = {
            str(k): int(v) for k, v in counts.items() if isinstance(v, (int, float))
        }

        created = 0
        candidates = list(reversed(ctx.all_events()[-scan_events:]))
        if scramble and len(candidates) > 1:
            ctx.rng.shuffle(candidates)
        for evt in candidates:
            etype = str(evt.get("type") or "")
            if not self._should_perceive(etype):
                continue
            sig = self._event_signature(evt)
            if sig in seen:
                continue

            prior = float(counts_map.get(etype, 0))
            novelty = 1.0 / (1.0 + prior)
            uncertainty = clamp01(1.0 - (prior / max(prior + 3.0, 1.0)), default=0.5)
            if noise_mag > 0.0:
                novelty = clamp01(novelty + ctx.rng.uniform(-noise_mag, noise_mag), default=novelty)
                uncertainty = clamp01(
                    uncertainty + ctx.rng.uniform(-(noise_mag * 0.5), noise_mag * 0.5),
                    default=uncertainty,
                )
            source = etype.split(".", 1)[0]

            percept_evt = ctx.emit_event(
                "sense.percept",
                {
                    "source_event_type": etype,
                    "source_module": source,
                    "novelty": round(novelty, 6),
                    "uncertainty": round(uncertainty, 6),
                    "strength": round(
                        clamp01(0.55 * novelty + 0.45 * uncertainty, default=0.5), 6
                    ),
                    "raw": dict((evt.get("data") or {})),
                },
                tags=["consciousness", "sense", "percept"],
                corr_id=str(evt.get("corr_id") or "") or None,
                parent_id=str(evt.get("parent_id") or "") or None,
            )

            if novelty >= max(emit_threshold, clamp_floor):
                payload = WorkspacePayload(
                    kind="PERCEPT",
                    source_module=self.name,
                    content={
                        "source_event_type": etype,
                        "novelty": round(novelty, 6),
                        "uncertainty": round(uncertainty, 6),
                    },
                    confidence=clamp01(1.0 - uncertainty, default=0.5),
                    salience=clamp01(novelty, default=0.5),
                    links={
                        "corr_id": percept_evt.get("corr_id"),
                        "parent_id": percept_evt.get("parent_id"),
                        "memory_ids": [],
                    },
                ).as_dict()
                payload = normalize_workspace_payload(
                    payload, fallback_kind="PERCEPT", source_module=self.name
                )
                ctx.broadcast(
                    self.name,
                    payload,
                    tags=["consciousness", "sense", "broadcast"],
                    corr_id=percept_evt.get("corr_id"),
                    parent_id=percept_evt.get("parent_id"),
                )

            counts_map[etype] = int(prior) + 1
            seen.add(sig)
            created += 1
            if created >= max_percepts:
                break

        state["event_type_counts"] = counts_map
        state["seen_signatures"] = list(seen)[-1200:]
        if created:
            ctx.metric("consciousness.sense.percepts", float(created))

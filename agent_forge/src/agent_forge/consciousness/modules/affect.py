from __future__ import annotations

from typing import Mapping

from ..types import TickContext
from ..types import WorkspacePayload, clamp01, normalize_workspace_payload


class AffectModule:
    name = "affect"

    def tick(self, ctx: TickContext) -> None:
        alpha = clamp01(ctx.config.get("affect_alpha"), default=0.25)
        emit_delta = clamp01(
            ctx.config.get("affect_emit_delta_threshold"), default=0.04
        )
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
        clamp_mag = max(
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
                "modulators": {
                    "arousal": 0.4,
                    "valence": 0.5,
                    "stability": 0.6,
                    "exploration_rate": 0.35,
                    "learning_rate": 0.25,
                    "attention_gain": 0.6,
                }
            },
        )

        intero_state = ctx.module_state("intero", defaults={"drives": {}})
        drives = (
            intero_state.get("drives")
            if isinstance(intero_state.get("drives"), Mapping)
            else {}
        )
        threat = clamp01(drives.get("threat"), default=0.25)
        curiosity = clamp01(drives.get("curiosity"), default=0.4)
        energy = clamp01(drives.get("energy"), default=0.5)
        coherence_hunger = clamp01(drives.get("coherence_hunger"), default=0.3)
        novelty_hunger = clamp01(drives.get("novelty_hunger"), default=0.35)

        prev = (
            state.get("modulators")
            if isinstance(state.get("modulators"), Mapping)
            else {}
        )
        prev_mod = {str(k): clamp01(v, default=0.5) for k, v in prev.items()}

        targets = {
            "arousal": clamp01((0.55 * threat) + (0.45 * curiosity), default=0.5),
            "valence": clamp01(
                (0.55 * energy) + (0.2 * curiosity) - (0.45 * threat), default=0.5
            ),
            "stability": clamp01(
                1.0 - (0.7 * threat) - (0.2 * novelty_hunger), default=0.5
            ),
            "exploration_rate": clamp01(
                (0.2 + (0.6 * curiosity) + (0.25 * novelty_hunger) - (0.4 * threat)),
                default=0.35,
            ),
            "learning_rate": clamp01(
                (0.15 + (0.4 * curiosity) + (0.3 * coherence_hunger)), default=0.25
            ),
            "attention_gain": clamp01(
                (0.45 + (0.45 * threat) + (0.25 * coherence_hunger)), default=0.6
            ),
        }
        if scramble:
            targets["exploration_rate"], targets["learning_rate"] = (
                targets["learning_rate"],
                targets["exploration_rate"],
            )
        if noise_mag > 0.0:
            for key, value in list(targets.items()):
                targets[key] = clamp01(value + ctx.rng.uniform(-noise_mag, noise_mag), default=value)
        if clamp_mag > 0.0:
            cap = max(0.0, 1.0 - clamp_mag)
            for key, value in list(targets.items()):
                targets[key] = min(cap, value)

        updated: dict[str, float] = {}
        max_delta = 0.0
        for key, target in targets.items():
            prev_value = prev_mod.get(key, 0.5)
            value = clamp01(
                (1.0 - alpha) * prev_value + (alpha * target), default=prev_value
            )
            updated[key] = round(value, 6)
            max_delta = max(max_delta, abs(value - prev_value))
            ctx.metric(f"consciousness.affect.{key}", float(updated[key]))

        state["modulators"] = updated

        emit = max_delta >= emit_delta
        if not emit:
            return

        evt = ctx.emit_event(
            "affect.modulators",
            {
                "modulators": updated,
                "targets": {k: round(v, 6) for k, v in targets.items()},
                "max_delta": round(max_delta, 6),
            },
            tags=["consciousness", "affect", "modulation"],
        )
        payload = WorkspacePayload(
            kind="AFFECT",
            source_module=self.name,
            content={
                "modulators": updated,
                "max_delta": round(max_delta, 6),
            },
            confidence=0.75,
            salience=clamp01(updated.get("attention_gain"), default=0.5),
            links={
                "corr_id": evt.get("corr_id"),
                "parent_id": evt.get("parent_id"),
                "memory_ids": [],
            },
        ).as_dict()
        payload = normalize_workspace_payload(
            payload, fallback_kind="AFFECT", source_module=self.name
        )
        ctx.broadcast(
            self.name,
            payload,
            tags=["consciousness", "affect", "broadcast"],
            corr_id=evt.get("corr_id"),
            parent_id=evt.get("parent_id"),
        )

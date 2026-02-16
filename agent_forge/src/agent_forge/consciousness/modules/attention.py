from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, Mapping

from ..types import TickContext, clamp01


def _event_signature(evt: Mapping[str, Any]) -> str:
    core = f"{evt.get('ts','')}|{evt.get('type','')}|{evt.get('corr_id','')}"
    return hashlib.sha1(core.encode("utf-8", "replace")).hexdigest()


def _guess_kind(event_type: str) -> str:
    if event_type.startswith("sense."):
        return "PERCEPT"
    if event_type.startswith("intero."):
        return "DRIVE"
    if event_type.startswith("mem."):
        return "MEMORY"
    if event_type.startswith("knowledge."):
        return "KNOWLEDGE"
    if event_type.startswith("world.prediction_error"):
        return "PRED_ERR"
    if event_type.startswith("policy."):
        return "PLAN"
    if event_type.startswith("self."):
        return "SELF"
    if event_type.startswith("meta."):
        return "META"
    if event_type.startswith("report."):
        return "REPORT"
    return "SIGNAL"


def _score_candidate(evt: Mapping[str, Any]) -> tuple[float, float]:
    etype = str(evt.get("type", ""))
    data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
    novelty = clamp01(data.get("novelty"), default=0.4)
    pred_err = clamp01(
        (
            abs(float(data.get("prediction_error", 0.0)))
            if data.get("prediction_error") is not None
            else 0.0
        ),
        default=0.0,
    )
    drive_strength = clamp01(
        (
            abs(float(data.get("strength", 0.0)))
            if data.get("strength") is not None
            else 0.0
        ),
        default=0.0,
    )

    if etype == "daemon.beat":
        base_salience = 0.2
        confidence = 0.8
    elif etype.startswith(("mem.", "knowledge.")):
        base_salience = clamp01(data.get("salience"), default=0.55)
        confidence = clamp01(data.get("confidence"), default=0.68)
    elif etype.startswith("workspace.broadcast"):
        payload = data.get("payload") if isinstance(data, Mapping) else {}
        base_salience = clamp01((payload or {}).get("salience"), default=0.45)
        confidence = clamp01((payload or {}).get("confidence"), default=0.7)
    else:
        base_salience = 0.3
        confidence = 0.65

    salience = clamp01(
        0.5 * base_salience + 0.2 * novelty + 0.2 * pred_err + 0.1 * drive_strength,
        default=base_salience,
    )
    return salience, confidence


def _working_set_boost(
    evt: Mapping[str, Any], active_items: list[Mapping[str, Any]]
) -> float:
    if not active_items:
        return 0.0
    etype = str(evt.get("type") or "")
    corr_id = str(evt.get("corr_id") or "")
    parent_id = str(evt.get("parent_id") or "")
    best = 0.0
    for item in active_items:
        salience = clamp01(item.get("salience"), default=0.0)
        signature = str(item.get("signature") or "")
        links = item.get("links") if isinstance(item.get("links"), Mapping) else {}
        item_corr = str(links.get("corr_id") or "")
        item_parent = str(links.get("parent_id") or "")
        if etype and signature and etype in signature:
            best = max(best, 0.5 * salience)
        if corr_id and (corr_id == item_corr or corr_id == item_parent):
            best = max(best, 0.85 * salience)
        if parent_id and (parent_id == item_corr or parent_id == item_parent):
            best = max(best, 0.75 * salience)
    return clamp01(best, default=0.0)


class AttentionModule:
    name = "attention"

    def __init__(self) -> None:
        self._seen_signatures: set[str] = set()

    def _prune_seen(self) -> None:
        # Keep bounded memory for long daemon runs.
        if len(self._seen_signatures) > 5000:
            self._seen_signatures = set(list(self._seen_signatures)[-2500:])

    def tick(self, ctx: TickContext) -> None:
        max_candidates = int(ctx.config.get("attention_max_candidates", 12))
        min_confidence = clamp01(
            ctx.config.get("attention_min_confidence"), default=0.2
        )
        ws_boost = clamp01(ctx.config.get("attention_working_set_boost"), default=0.12)
        created = 0
        ws_state = ctx.module_state("working_set", defaults={"active_items": []})
        raw_active = ws_state.get("active_items")
        active_items = list(raw_active) if isinstance(raw_active, list) else []
        affect_state = ctx.module_state("affect", defaults={"modulators": {}})
        modulators = (
            affect_state.get("modulators")
            if isinstance(affect_state.get("modulators"), Mapping)
            else {}
        )
        attention_gain = clamp01(modulators.get("attention_gain"), default=0.6)
        exploration_rate = clamp01(modulators.get("exploration_rate"), default=0.35)

        perturbations = ctx.perturbations_for(self.name)
        drop_active = any(p.get("kind") == "drop" for p in perturbations)
        scramble_active = any(p.get("kind") == "scramble" for p in perturbations)
        delay_active = any(p.get("kind") == "delay" for p in perturbations)
        clamp_ceiling = 1.0
        perturb_noise = 0.0
        for pert in perturbations:
            kind = str(pert.get("kind") or "")
            magnitude = clamp01(pert.get("magnitude"), default=0.0)
            if kind == "noise":
                perturb_noise = max(perturb_noise, magnitude)
            elif kind == "clamp":
                clamp_ceiling = min(clamp_ceiling, max(0.05, magnitude))

        if drop_active:
            ctx.metric("consciousness.attention.candidates", 0.0)
            return
        if delay_active and (ctx.beat_count % 2 == 1):
            return

        # Prioritize fresh events emitted by earlier modules in this same tick.
        event_stream = list(ctx.recent_events) + list(ctx.emitted_events)
        if scramble_active:
            event_stream = list(event_stream)
            ctx.rng.shuffle(event_stream)
        for evt in reversed(event_stream):
            etype = str(evt.get("type") or "")
            if not etype or etype.startswith(("attn.", "gw.", "wm.")):
                continue

            sig = _event_signature(evt)
            if sig in self._seen_signatures:
                continue

            salience, confidence = _score_candidate(evt)
            relevance = _working_set_boost(evt, active_items)
            salience = clamp01(salience + (ws_boost * relevance), default=salience)
            salience = clamp01(
                salience * (0.75 + (0.5 * attention_gain)), default=salience
            )
            confidence = clamp01(
                confidence * (0.8 + (0.3 * attention_gain)), default=confidence
            )

            noise = float(ctx.config.get("attention_score_noise", 0.0))
            noise += perturb_noise * max(0.05, exploration_rate)
            if noise > 0.0:
                salience = clamp01(
                    salience + ctx.rng.uniform(-noise, noise), default=salience
                )
            salience = min(salience, clamp_ceiling)
            if confidence < min_confidence:
                continue
            candidate_id = uuid.uuid4().hex
            score = clamp01((0.68 * salience) + (0.32 * confidence), default=0.0)
            data: Dict[str, Any] = {
                "candidate_id": candidate_id,
                "source_event_type": etype,
                "source_module": str(
                    (evt.get("data") or {}).get("source") or etype.split(".", 1)[0]
                ),
                "kind": _guess_kind(etype),
                "salience": salience,
                "confidence": confidence,
                "score": round(score, 4),
                "links": {
                    "corr_id": str(evt.get("corr_id") or ""),
                    "parent_id": str(evt.get("parent_id") or ""),
                    "memory_ids": [],
                },
                "content": {
                    "event_ts": evt.get("ts"),
                    "event_type": etype,
                    "event_data": dict((evt.get("data") or {})),
                    "working_set_relevance": round(relevance, 6),
                    "attention_gain": round(attention_gain, 6),
                    "exploration_rate": round(exploration_rate, 6),
                },
            }
            ctx.emit_event("attn.candidate", data, tags=["consciousness", "attention"])
            self._seen_signatures.add(sig)
            created += 1
            if created >= max_candidates:
                break

        if created:
            ctx.metric("consciousness.attention.candidates", float(created))
        self._prune_seen()

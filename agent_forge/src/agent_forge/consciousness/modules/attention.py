from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, Mapping, Sequence

from ..types import TickContext, clamp01


def _event_signature(evt: Mapping[str, Any]) -> str:
    core = f"{evt.get('ts','')}|{evt.get('type','')}|{evt.get('corr_id','')}"
    return hashlib.sha1(core.encode("utf-8", "replace")).hexdigest()


def _guess_kind(event_type: str) -> str:
    if event_type.startswith("sense."):
        return "PERCEPT"
    if event_type.startswith("intero."):
        return "DRIVE"
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
    pred_err = clamp01(abs(float(data.get("prediction_error", 0.0))) if data.get("prediction_error") is not None else 0.0, default=0.0)
    drive_strength = clamp01(abs(float(data.get("strength", 0.0))) if data.get("strength") is not None else 0.0, default=0.0)

    if etype == "daemon.beat":
        base_salience = 0.2
        confidence = 0.8
    elif etype.startswith("workspace.broadcast"):
        payload = data.get("payload") if isinstance(data, Mapping) else {}
        base_salience = clamp01((payload or {}).get("salience"), default=0.45)
        confidence = clamp01((payload or {}).get("confidence"), default=0.7)
    else:
        base_salience = 0.3
        confidence = 0.65

    salience = clamp01(0.5 * base_salience + 0.2 * novelty + 0.2 * pred_err + 0.1 * drive_strength, default=base_salience)
    return salience, confidence


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
        created = 0
        for evt in reversed(ctx.recent_events):
            etype = str(evt.get("type") or "")
            if not etype or etype.startswith("attn.") or etype.startswith("gw."):
                continue

            sig = _event_signature(evt)
            if sig in self._seen_signatures:
                continue

            salience, confidence = _score_candidate(evt)
            noise = float(ctx.config.get("attention_score_noise", 0.0))
            if noise > 0.0:
                salience = clamp01(salience + ctx.rng.uniform(-noise, noise), default=salience)
            candidate_id = uuid.uuid4().hex
            data: Dict[str, Any] = {
                "candidate_id": candidate_id,
                "source_event_type": etype,
                "source_module": str((evt.get("data") or {}).get("source") or etype.split(".", 1)[0]),
                "kind": _guess_kind(etype),
                "salience": salience,
                "confidence": confidence,
                "score": round(0.7 * salience + 0.3 * confidence, 4),
                "links": {
                    "corr_id": str(evt.get("corr_id") or ""),
                    "parent_id": str(evt.get("parent_id") or ""),
                    "memory_ids": [],
                },
                "content": {
                    "event_ts": evt.get("ts"),
                    "event_type": etype,
                    "event_data": dict((evt.get("data") or {})),
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

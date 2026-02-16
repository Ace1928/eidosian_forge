from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Mapping

from ..types import TickContext, WorkspacePayload, normalize_workspace_payload


def _event_signature(evt: Mapping[str, Any]) -> str:
    return f"{evt.get('ts','')}|{evt.get('type','')}|{evt.get('corr_id','')}"


def _should_model(etype: str) -> bool:
    if not etype:
        return False
    if etype.startswith(("world.", "metrics.", "attn.", "gw.")):
        return False
    return True


class WorldModelModule:
    name = "world_model"

    def __init__(self) -> None:
        self._seen_signatures: set[str] = set()
        self._transitions: dict[str, Counter[str]] = defaultdict(Counter)
        self._last_event_type: str | None = None

    def _predict(self, prev_event_type: str, current_event_type: str) -> Dict[str, Any]:
        dist = self._transitions.get(prev_event_type, Counter())
        total = sum(dist.values())
        if total <= 0:
            return {
                "predicted_next_event_type": None,
                "actual_event_type": current_event_type,
                "probability_actual": 0.0,
                "prediction_error": 1.0,
                "known_transition_count": 0,
            }

        predicted_next_event_type = max(dist.items(), key=lambda item: item[1])[0]
        probability_actual = float(dist.get(current_event_type, 0) / max(total, 1))
        prediction_error = 1.0 - probability_actual
        return {
            "predicted_next_event_type": predicted_next_event_type,
            "actual_event_type": current_event_type,
            "probability_actual": round(probability_actual, 6),
            "prediction_error": round(prediction_error, 6),
            "known_transition_count": int(total),
        }

    def tick(self, ctx: TickContext) -> None:
        threshold = float(ctx.config.get("world_error_broadcast_threshold", 0.55))
        max_updates = int(ctx.config.get("world_prediction_window", 120))
        processed = 0

        candidates = list(ctx.recent_events) + list(ctx.emitted_events)
        for evt in candidates:
            etype = str(evt.get("type") or "")
            if not _should_model(etype):
                continue
            sig = _event_signature(evt)
            if sig in self._seen_signatures:
                continue

            if self._last_event_type:
                prediction = self._predict(self._last_event_type, etype)
                corr_id = str(evt.get("corr_id") or "")
                parent_id = str(evt.get("parent_id") or "")

                pred_evt = ctx.emit_event(
                    "world.prediction",
                    {
                        "context_event_type": self._last_event_type,
                        **prediction,
                    },
                    tags=["consciousness", "world_model", "prediction"],
                    corr_id=corr_id or None,
                    parent_id=parent_id or None,
                )

                err = float(prediction.get("prediction_error") or 1.0)
                err_evt = ctx.emit_event(
                    "world.prediction_error",
                    {
                        "context_event_type": self._last_event_type,
                        "actual_event_type": etype,
                        "predicted_event_type": prediction.get("predicted_next_event_type"),
                        "prediction_error": err,
                        "probability_actual": float(prediction.get("probability_actual") or 0.0),
                        "known_transition_count": int(prediction.get("known_transition_count") or 0),
                        "unexpected": bool(prediction.get("predicted_next_event_type"))
                        and prediction.get("predicted_next_event_type") != etype,
                    },
                    tags=["consciousness", "world_model", "prediction_error"],
                    corr_id=pred_evt.get("corr_id"),
                    parent_id=pred_evt.get("parent_id"),
                )
                ctx.metric("consciousness.world.prediction_error", err)

                if err >= threshold:
                    payload = WorkspacePayload(
                        kind="PRED_ERR",
                        source_module="world_model",
                        content={
                            "context_event_type": self._last_event_type,
                            "actual_event_type": etype,
                            "predicted_event_type": prediction.get("predicted_next_event_type"),
                            "prediction_error": err,
                            "known_transition_count": int(prediction.get("known_transition_count") or 0),
                        },
                        confidence=max(0.0, min(1.0, 1.0 - err)),
                        salience=max(0.0, min(1.0, err)),
                        links={
                            "corr_id": err_evt.get("corr_id"),
                            "parent_id": err_evt.get("parent_id"),
                            "memory_ids": [],
                        },
                    ).as_dict()
                    payload = normalize_workspace_payload(
                        payload,
                        fallback_kind="PRED_ERR",
                        source_module="world_model",
                    )
                    ctx.broadcast(
                        "world_model",
                        payload,
                        tags=["consciousness", "world_model", "prediction_error", "broadcast"],
                        corr_id=err_evt.get("corr_id"),
                        parent_id=err_evt.get("parent_id"),
                    )

            if self._last_event_type:
                self._transitions[self._last_event_type][etype] += 1
            self._last_event_type = etype
            self._seen_signatures.add(sig)
            processed += 1
            if processed >= max_updates:
                break

        if processed:
            ctx.metric("consciousness.world.updates", float(processed))

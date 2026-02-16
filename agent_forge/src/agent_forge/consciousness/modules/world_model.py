from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Mapping

from ..features import extract_event_features
from ..types import TickContext, WorkspacePayload, clamp01, normalize_workspace_payload


def _event_signature(evt: Mapping[str, Any]) -> str:
    return f"{evt.get('ts','')}|{evt.get('type','')}|{evt.get('corr_id','')}"


def _should_model(etype: str) -> bool:
    if not etype:
        return False
    if etype.startswith(("world.", "metrics.", "attn.", "gw.")):
        return False
    return True


def _top_items(mapping: Mapping[str, float], k: int = 8) -> list[dict[str, Any]]:
    rows = sorted(mapping.items(), key=lambda item: abs(float(item[1])), reverse=True)
    out: list[dict[str, Any]] = []
    for key, value in rows[: max(1, int(k))]:
        out.append({"feature": str(key), "value": round(float(value), 6)})
    return out


def _merge_belief(
    belief: dict[str, float],
    features: Mapping[str, float],
    *,
    alpha: float,
    max_features: int,
) -> dict[str, float]:
    alpha = max(0.01, min(0.95, float(alpha)))
    merged: dict[str, float] = dict(belief)
    keys = set(merged.keys()) | set(features.keys())
    for key in keys:
        old = float(merged.get(key, 0.0))
        cur = float(features.get(key, 0.0))
        updated = ((1.0 - alpha) * old) + (alpha * cur)
        if abs(updated) >= 1e-5:
            merged[key] = updated
        elif key in merged:
            merged.pop(key, None)

    if len(merged) > max(1, int(max_features)):
        sorted_keys = sorted(merged.keys(), key=lambda k: abs(merged[k]), reverse=True)
        keep = set(sorted_keys[: max(1, int(max_features))])
        merged = {k: merged[k] for k in keep}
    return merged


def _feature_error(predicted: Mapping[str, float], actual: Mapping[str, float]) -> tuple[float, dict[str, float]]:
    keys = set(predicted.keys()) | set(actual.keys())
    if not keys:
        return 0.0, {}
    diff: dict[str, float] = {}
    for key in keys:
        diff[key] = abs(float(actual.get(key, 0.0)) - float(predicted.get(key, 0.0)))
    err = sum(diff.values()) / max(len(diff), 1)
    return float(max(0.0, min(1.0, err))), diff


class WorldModelModule:
    name = "world_model"

    def __init__(self) -> None:
        self._seen_signatures: set[str] = set()
        self._transitions: dict[str, Counter[str]] = defaultdict(Counter)
        self._event_counts: Counter[str] = Counter()
        self._last_event_type: str | None = None
        self._belief_cache: dict[str, float] = {}

    def rollout(self, steps: int = 3, policy_hint: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        steps = max(0, int(steps))
        if steps <= 0:
            return []
        start = str((policy_hint or {}).get("start_event_type") or self._last_event_type or "")
        current = start
        if not current and self._event_counts:
            current = max(self._event_counts.items(), key=lambda item: item[1])[0]

        out: list[dict[str, Any]] = []
        for i in range(steps):
            dist = self._transitions.get(current or "", Counter())
            if dist:
                next_type = max(dist.items(), key=lambda item: item[1])[0]
                confidence = float(max(dist.values()) / max(sum(dist.values()), 1))
            else:
                next_type = max(self._event_counts.items(), key=lambda item: item[1])[0] if self._event_counts else "unknown"
                confidence = 0.0
            out.append(
                {
                    "step": i + 1,
                    "context_event_type": current or None,
                    "predicted_event_type": next_type,
                    "confidence": round(float(max(0.0, min(1.0, confidence))), 6),
                    "belief_top_features": _top_items(self._belief_cache, 6),
                }
            )
            current = next_type
        return out

    def _predict_transition(self, prev_event_type: str, current_event_type: str) -> Dict[str, Any]:
        dist = self._transitions.get(prev_event_type, Counter())
        total = sum(dist.values())
        if total <= 0:
            return {
                "predicted_next_event_type": None,
                "actual_event_type": current_event_type,
                "probability_actual": 0.0,
                "transition_prediction_error": 1.0,
                "known_transition_count": 0,
            }

        predicted_next_event_type = max(dist.items(), key=lambda item: item[1])[0]
        probability_actual = float(dist.get(current_event_type, 0) / max(total, 1))
        prediction_error = 1.0 - probability_actual
        return {
            "predicted_next_event_type": predicted_next_event_type,
            "actual_event_type": current_event_type,
            "probability_actual": round(probability_actual, 6),
            "transition_prediction_error": round(prediction_error, 6),
            "known_transition_count": int(total),
        }

    def tick(self, ctx: TickContext) -> None:
        threshold = float(ctx.config.get("world_error_broadcast_threshold", 0.55))
        derivative_threshold = float(ctx.config.get("world_error_derivative_threshold", 0.2))
        max_updates = int(ctx.config.get("world_prediction_window", 120))
        alpha = clamp01(ctx.config.get("world_belief_alpha"), default=0.22)
        top_k = max(1, int(ctx.config.get("world_belief_top_k", 8)))
        max_features = max(32, int(ctx.config.get("world_belief_max_features", 256)))
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
        if clamp_mag > 0.0:
            threshold = min(1.0, threshold + (clamp_mag * 0.25))
            derivative_threshold = min(1.0, derivative_threshold + (clamp_mag * 0.15))

        state = ctx.module_state(
            self.name,
            defaults={"belief": {}, "last_prediction_error": None, "total_updates": 0},
        )
        belief = {
            str(k): float(v)
            for k, v in dict(state.get("belief") or {}).items()
            if isinstance(v, (int, float))
        }

        processed = 0
        last_error: float | None = (
            float(state.get("last_prediction_error"))
            if isinstance(state.get("last_prediction_error"), (int, float))
            else None
        )
        latest_pred_evt: dict[str, Any] | None = None
        latest_err_evt: dict[str, Any] | None = None

        candidates = list(ctx.recent_events) + list(ctx.emitted_events)
        if scramble and len(candidates) > 1:
            ctx.rng.shuffle(candidates)
        for evt in candidates:
            etype = str(evt.get("type") or "")
            if not _should_model(etype):
                continue
            sig = _event_signature(evt)
            if sig in self._seen_signatures:
                continue

            features = extract_event_features(evt)
            if noise_mag > 0.0 and features:
                noisy: dict[str, float] = {}
                for key, value in features.items():
                    noisy[key] = float(value) + ctx.rng.uniform(-noise_mag, noise_mag)
                features = noisy
            predicted_features = dict(belief)
            feature_error, feature_diff = _feature_error(predicted_features, features)
            if noise_mag > 0.0:
                feature_error = clamp01(
                    feature_error + ctx.rng.uniform(-(noise_mag * 0.25), noise_mag * 0.35),
                    default=feature_error,
                )
            top_feature_errors = _top_items(feature_diff, top_k)

            transition = {
                "predicted_next_event_type": None,
                "probability_actual": 0.0,
                "transition_prediction_error": 1.0,
                "known_transition_count": 0,
            }
            if self._last_event_type:
                transition = self._predict_transition(self._last_event_type, etype)

            corr_id = str(evt.get("corr_id") or "")
            parent_id = str(evt.get("parent_id") or "")
            latest_pred_evt = ctx.emit_event(
                "world.prediction",
                {
                    "context_event_type": self._last_event_type,
                    "predicted_next_event_type": transition.get("predicted_next_event_type"),
                    "actual_event_type": etype,
                    "probability_actual": float(transition.get("probability_actual") or 0.0),
                    "transition_prediction_error": float(transition.get("transition_prediction_error") or 1.0),
                    "prediction_error": float(feature_error),
                    "predicted_feature_count": len(predicted_features),
                    "predicted_features_top": _top_items(predicted_features, top_k),
                    "known_transition_count": int(transition.get("known_transition_count") or 0),
                },
                tags=["consciousness", "world_model", "prediction"],
                corr_id=corr_id or None,
                parent_id=parent_id or None,
            )

            err_derivative = 0.0
            if last_error is not None:
                err_derivative = float(feature_error - last_error)
            latest_err_evt = ctx.emit_event(
                "world.prediction_error",
                {
                    "context_event_type": self._last_event_type,
                    "actual_event_type": etype,
                    "predicted_event_type": transition.get("predicted_next_event_type"),
                    "prediction_error": float(feature_error),
                    "prediction_error_derivative": round(float(err_derivative), 6),
                    "probability_actual": float(transition.get("probability_actual") or 0.0),
                    "known_transition_count": int(transition.get("known_transition_count") or 0),
                    "unexpected": bool(transition.get("predicted_next_event_type"))
                    and transition.get("predicted_next_event_type") != etype,
                    "top_feature_errors": top_feature_errors,
                },
                tags=["consciousness", "world_model", "prediction_error"],
                corr_id=latest_pred_evt.get("corr_id"),
                parent_id=latest_pred_evt.get("parent_id"),
            )
            ctx.metric("consciousness.world.prediction_error", float(feature_error))

            if feature_error >= threshold or err_derivative >= derivative_threshold:
                payload = WorkspacePayload(
                    kind="PRED_ERR",
                    source_module="world_model",
                    content={
                        "context_event_type": self._last_event_type,
                        "actual_event_type": etype,
                        "predicted_event_type": transition.get("predicted_next_event_type"),
                        "prediction_error": float(feature_error),
                        "prediction_error_derivative": round(float(err_derivative), 6),
                        "top_feature_errors": top_feature_errors,
                        "known_transition_count": int(transition.get("known_transition_count") or 0),
                    },
                    confidence=max(0.0, min(1.0, 1.0 - feature_error)),
                    salience=max(0.0, min(1.0, feature_error)),
                    links={
                        "corr_id": latest_err_evt.get("corr_id"),
                        "parent_id": latest_err_evt.get("parent_id"),
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
                    corr_id=latest_err_evt.get("corr_id"),
                    parent_id=latest_err_evt.get("parent_id"),
                )

            belief = _merge_belief(
                belief,
                features,
                alpha=alpha,
                max_features=max_features,
            )

            if self._last_event_type:
                self._transitions[self._last_event_type][etype] += 1
            self._event_counts[etype] += 1
            self._last_event_type = etype
            self._seen_signatures.add(sig)
            last_error = float(feature_error)
            processed += 1
            if processed >= max_updates:
                break

        if processed and latest_err_evt is not None:
            belief_top = _top_items(belief, top_k)
            belief_evt = ctx.emit_event(
                "world.belief_state",
                {
                    "feature_count": len(belief),
                    "belief_top_features": belief_top,
                    "last_prediction_error": float(last_error or 0.0),
                    "rollout_preview": self.rollout(
                        steps=max(1, int(ctx.config.get("world_rollout_default_steps", 3)))
                    ),
                },
                tags=["consciousness", "world_model", "belief_state"],
                corr_id=latest_err_evt.get("corr_id"),
                parent_id=latest_err_evt.get("parent_id"),
            )
            ctx.metric("consciousness.world.updates", float(processed))
            ctx.metric("consciousness.world.feature_count", float(len(belief)))

            self._belief_cache = dict(belief)
            state["belief"] = {k: round(float(v), 8) for k, v in belief.items()}
            state["last_prediction_error"] = float(last_error or 0.0)
            state["total_updates"] = int(state.get("total_updates") or 0) + processed
            state["last_belief_corr_id"] = belief_evt.get("corr_id")
        elif processed:
            ctx.metric("consciousness.world.updates", float(processed))

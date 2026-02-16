from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping

from .params import ParamSpec, default_param_specs


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _get_nested(payload: Mapping[str, Any], key_path: str) -> Any:
    node: Any = payload
    for part in str(key_path).split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node


def _set_nested(payload: MutableMapping[str, Any], key_path: str, value: Any) -> None:
    parts = [p for p in str(key_path).split(".") if p]
    if not parts:
        return
    node: MutableMapping[str, Any] = payload
    for part in parts[:-1]:
        existing = node.get(part)
        if not isinstance(existing, MutableMapping):
            existing = {}
            node[part] = existing
        node = existing
    node[parts[-1]] = value


def _coerce_to_spec(value: Any, spec: ParamSpec) -> Any:
    if spec.kind == "bool":
        return bool(value)
    if spec.kind == "int":
        v = _safe_int(value, default=int(spec.default))
        if spec.min_value is not None:
            v = max(int(spec.min_value), v)
        if spec.max_value is not None:
            v = min(int(spec.max_value), v)
        return v
    if spec.kind == "float":
        v = _safe_float(value, default=float(spec.default))
        if spec.min_value is not None:
            v = max(float(spec.min_value), v)
        if spec.max_value is not None:
            v = min(float(spec.max_value), v)
        return round(v, 6)
    if spec.kind == "choice":
        choices = list(spec.choices or [])
        if value in choices:
            return value
        return spec.default
    return value


def sanitize_overlay(
    overlay: Mapping[str, Any] | None,
    *,
    specs: Mapping[str, ParamSpec] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    specs = dict(specs or default_param_specs())
    cleaned: dict[str, Any] = {}
    invalid: list[str] = []
    if not isinstance(overlay, Mapping):
        return cleaned, invalid

    for key, spec in specs.items():
        raw = _get_nested(overlay, key)
        if raw is None:
            continue
        _set_nested(cleaned, key, _coerce_to_spec(raw, spec))

    def _flatten_unknown(node: Any, prefix: str = "") -> None:
        if not isinstance(node, Mapping):
            if prefix and prefix not in specs:
                invalid.append(prefix)
            return
        for child_key, child_value in node.items():
            child = f"{prefix}.{child_key}" if prefix else str(child_key)
            if isinstance(child_value, Mapping):
                _flatten_unknown(child_value, child)
                continue
            if child not in specs:
                invalid.append(child)

    _flatten_unknown(dict(overlay))
    invalid = sorted(set(invalid))
    return cleaned, invalid


def apply_overlay(
    base_config: Mapping[str, Any],
    overlay: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base_config))
    if not isinstance(overlay, Mapping):
        return merged

    def _merge(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> None:
        for key, value in src.items():
            if isinstance(value, Mapping):
                existing = dst.get(key)
                if not isinstance(existing, MutableMapping):
                    existing = {}
                    dst[key] = existing
                _merge(existing, value)
                continue
            dst[key] = value

    _merge(merged, dict(overlay))
    return merged


def resolve_config(
    base_config: Mapping[str, Any],
    *,
    tuned_overlay: Mapping[str, Any] | None = None,
    runtime_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = apply_overlay(base_config, tuned_overlay)
    cfg = apply_overlay(cfg, runtime_overrides)
    return cfg


def load_tuned_overlay(state_store: Any) -> tuple[dict[str, Any], list[str]]:
    if state_store is None:
        return {}, []
    raw = state_store.get_meta("tuned_overlay", {})
    return sanitize_overlay(raw)


def persist_tuned_overlay(
    state_store: Any,
    overlay: Mapping[str, Any],
    *,
    source: str,
    reason: str,
    score: float | None = None,
) -> dict[str, Any]:
    cleaned, invalid = sanitize_overlay(overlay)
    if state_store is None:
        return {"overlay": cleaned, "invalid_keys": invalid, "version": 0}

    version = int(state_store.get_meta("tuned_overlay_version", 0) or 0) + 1
    history = state_store.get_meta("tuned_overlay_history", [])
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "version": version,
            "ts": _now_iso(),
            "source": str(source or "unknown"),
            "reason": str(reason or ""),
            "score": float(score) if score is not None else None,
            "overlay": cleaned,
            "invalid_keys": invalid,
        }
    )
    state_store.set_meta("tuned_overlay", cleaned)
    state_store.set_meta("tuned_overlay_version", version)
    state_store.set_meta("tuned_overlay_history", history[-60:])
    state_store.mark_dirty()
    return {"overlay": cleaned, "invalid_keys": invalid, "version": version}


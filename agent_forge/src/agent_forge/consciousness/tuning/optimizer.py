from __future__ import annotations

import copy
from typing import Any, Mapping, MutableMapping

from .params import ParamSpec


def _flatten_overlay(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten_overlay(value, path))
            continue
        out[path] = value
    return out


def _set_nested(payload: MutableMapping[str, Any], key_path: str, value: Any) -> None:
    parts = [p for p in str(key_path).split(".") if p]
    if not parts:
        return
    node: MutableMapping[str, Any] = payload
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


class BanditOptimizer:
    """
    Lightweight hill-climbing bandit optimizer.

    It mutates one tuned parameter per proposal and tracks rolling acceptance state.
    """

    def __init__(
        self,
        *,
        param_specs: Mapping[str, ParamSpec],
        state: MutableMapping[str, Any],
        rng: Any,
    ) -> None:
        self.param_specs = dict(param_specs)
        self.state = state
        self.rng = rng
        self.state.setdefault("attempts", 0)
        self.state.setdefault("accepted", 0)
        self.state.setdefault("last_key", "")
        self.state.setdefault("step_scale", 0.2)

    def _pick_key(self) -> str:
        keys = sorted(self.param_specs.keys())
        if not keys:
            raise ValueError("no parameter specs configured")
        return str(self.rng.choice(keys))

    def _mutate(self, current: Any, spec: ParamSpec) -> Any:
        if spec.kind == "bool":
            return not bool(current)
        if spec.kind == "choice":
            choices = list(spec.choices or [])
            if not choices:
                return spec.default
            current_idx = choices.index(current) if current in choices else 0
            step = 1 if self.rng.random() > 0.5 else -1
            return choices[(current_idx + step) % len(choices)]
        if spec.kind in {"int", "float"}:
            lo = spec.min_value if spec.min_value is not None else spec.default
            hi = spec.max_value if spec.max_value is not None else spec.default
            lo_f = float(lo)
            hi_f = float(hi)
            span = max(1e-9, hi_f - lo_f)
            scale = float(self.state.get("step_scale") or 0.2)
            direction = -1.0 if self.rng.random() < 0.5 else 1.0
            delta = direction * span * scale * (0.25 + (0.75 * self.rng.random()))
            base = float(current if current is not None else spec.default)
            nxt = min(hi_f, max(lo_f, base + delta))
            if spec.kind == "int":
                return int(round(nxt))
            return round(float(nxt), 6)
        return current

    def propose(self, current_overlay: Mapping[str, Any]) -> dict[str, Any]:
        overlay = copy.deepcopy(dict(current_overlay))
        flat = _flatten_overlay(overlay)
        key = self._pick_key()
        spec = self.param_specs[key]
        current_value = flat.get(key, spec.default)
        candidate_value = self._mutate(current_value, spec)
        _set_nested(overlay, key, candidate_value)
        self.state["attempts"] = int(self.state.get("attempts") or 0) + 1
        self.state["last_key"] = key
        return {
            "overlay": overlay,
            "key": key,
            "before": current_value,
            "after": candidate_value,
        }

    def observe(self, *, accepted: bool) -> None:
        if accepted:
            self.state["accepted"] = int(self.state.get("accepted") or 0) + 1
            scale = float(self.state.get("step_scale") or 0.2)
            self.state["step_scale"] = min(0.35, scale * 1.05)
            return
        scale = float(self.state.get("step_scale") or 0.2)
        self.state["step_scale"] = max(0.04, scale * 0.95)

    def observe_result(
        self,
        *,
        accepted: bool,
        score: float | None = None,
        overlay: Mapping[str, Any] | None = None,
        objectives: Mapping[str, float] | None = None,
        report: Mapping[str, Any] | None = None,
    ) -> None:
        # Compatibility hook for shared optimizer interface.
        self.observe(accepted=accepted)

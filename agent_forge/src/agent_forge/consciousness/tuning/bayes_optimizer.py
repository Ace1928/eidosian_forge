from __future__ import annotations

import copy
import math
from typing import Any, Mapping, MutableMapping

from .optimizer import _flatten_overlay, _set_nested
from .params import ParamSpec


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_objectives(raw: Mapping[str, Any] | None) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _distance(a: Mapping[str, Any], b: Mapping[str, Any], specs: Mapping[str, ParamSpec]) -> float:
    total = 0.0
    for key, spec in specs.items():
        av = a.get(key, spec.default)
        bv = b.get(key, spec.default)
        if spec.kind in {"int", "float"}:
            lo = _safe_float(spec.min_value, _safe_float(spec.default))
            hi = _safe_float(spec.max_value, _safe_float(spec.default))
            span = max(1e-6, hi - lo)
            da = (_safe_float(av, _safe_float(spec.default)) - _safe_float(bv, _safe_float(spec.default))) / span
            total += da * da
            continue
        if spec.kind == "bool":
            total += 0.0 if bool(av) == bool(bv) else 1.0
            continue
        if spec.kind == "choice":
            total += 0.0 if av == bv else 1.0
            continue
    return math.sqrt(total)


def _dominates(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    shared = set(a.keys()) & set(b.keys())
    if not shared:
        return False
    ge_all = all(_safe_float(a.get(k), 0.0) >= _safe_float(b.get(k), 0.0) for k in shared)
    gt_one = any(_safe_float(a.get(k), 0.0) > _safe_float(b.get(k), 0.0) for k in shared)
    return ge_all and gt_one


def pareto_front(points: list[dict[str, float]]) -> list[dict[str, float]]:
    if not points:
        return []
    front: list[dict[str, float]] = []
    for idx, point in enumerate(points):
        dominated = False
        for jdx, other in enumerate(points):
            if idx == jdx:
                continue
            if _dominates(other, point):
                dominated = True
                break
        if not dominated:
            front.append(point)
    return front


class BayesParetoOptimizer:
    """
    Lightweight Bayesian-style optimizer with multi-objective frontier tracking.

    Surrogate behavior:
    - Uses RBF-like weighting over historical observations.
    - Estimates expected score and objective vectors.
    - Chooses candidate by acquisition = mean + kappa*uncertainty + diversity + pareto_bonus.
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
        self.state.setdefault("optimizer_kind", "bayes_pareto")
        self.state.setdefault("attempts", 0)
        self.state.setdefault("accepted", 0)
        self.state.setdefault("history", [])
        self.state.setdefault("candidate_pool", 14)
        self.state.setdefault("kernel_gamma", 3.5)
        self.state.setdefault("kappa", 0.35)
        self.state.setdefault("exploration", 0.12)

    def _history(self) -> list[dict[str, Any]]:
        raw = self.state.get("history")
        if not isinstance(raw, list):
            raw = []
            self.state["history"] = raw
        return [row for row in raw if isinstance(row, Mapping)]

    def _mutate(self, current: Any, spec: ParamSpec) -> Any:
        if spec.kind == "bool":
            return not bool(current)
        if spec.kind == "choice":
            choices = list(spec.choices or [])
            if not choices:
                return spec.default
            if current not in choices:
                return choices[0]
            current_idx = choices.index(current)
            step = 1 if self.rng.random() > 0.5 else -1
            return choices[(current_idx + step) % len(choices)]
        if spec.kind in {"int", "float"}:
            lo = _safe_float(spec.min_value, _safe_float(spec.default))
            hi = _safe_float(spec.max_value, _safe_float(spec.default))
            span = max(1e-9, hi - lo)
            exploration = _safe_float(self.state.get("exploration"), 0.12)
            delta = span * exploration * (0.4 + 1.6 * self.rng.random())
            direction = -1.0 if self.rng.random() < 0.5 else 1.0
            nxt = _safe_float(current, _safe_float(spec.default)) + (direction * delta)
            nxt = min(hi, max(lo, nxt))
            if spec.kind == "int":
                return int(round(nxt))
            return round(float(nxt), 6)
        return current

    def _random_candidate(self, overlay: Mapping[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(dict(overlay))
        flat = _flatten_overlay(out)
        key = str(self.rng.choice(sorted(self.param_specs.keys())))
        spec = self.param_specs[key]
        current_value = flat.get(key, spec.default)
        _set_nested(out, key, self._mutate(current_value, spec))
        return out

    def _surrogate(
        self, candidate_flat: Mapping[str, Any], history: list[dict[str, Any]]
    ) -> tuple[float, float, dict[str, float]]:
        if not history:
            return 0.0, 1.0, {}
        gamma = max(0.1, _safe_float(self.state.get("kernel_gamma"), 3.5))
        weights: list[float] = []
        scores: list[float] = []
        objective_keys: set[str] = set()
        for row in history:
            obj = row.get("objectives")
            if isinstance(obj, Mapping):
                objective_keys.update(str(k) for k in obj.keys())

        objective_values: dict[str, list[float]] = {k: [] for k in objective_keys}
        for row in history:
            flat = row.get("flat_overlay") if isinstance(row.get("flat_overlay"), Mapping) else {}
            dist = _distance(candidate_flat, flat, self.param_specs)
            w = math.exp(-gamma * dist * dist)
            weights.append(w)
            scores.append(_safe_float(row.get("score"), 0.0))
            row_obj = _parse_objectives(row.get("objectives") if isinstance(row.get("objectives"), Mapping) else {})
            for key in objective_keys:
                objective_values[key].append(_safe_float(row_obj.get(key), 0.0))

        w_sum = sum(weights)
        if w_sum <= 1e-9:
            return 0.0, 1.0, {k: 0.0 for k in objective_keys}

        mean = sum(w * s for w, s in zip(weights, scores)) / w_sum
        second = sum(w * (s * s) for w, s in zip(weights, scores)) / w_sum
        var = max(1e-9, second - (mean * mean))
        uncertainty = math.sqrt(var)
        predicted_objectives = {
            key: sum(w * v for w, v in zip(weights, vals)) / w_sum for key, vals in objective_values.items()
        }
        return mean, uncertainty, predicted_objectives

    def propose(self, current_overlay: Mapping[str, Any]) -> dict[str, Any]:
        history = self._history()
        candidate_pool = max(4, int(self.state.get("candidate_pool") or 14))
        kappa = max(0.0, _safe_float(self.state.get("kappa"), 0.35))

        best_score_hist = max([_safe_float(row.get("score"), 0.0) for row in history] or [0.0])
        seen_flats = [row.get("flat_overlay") for row in history if isinstance(row.get("flat_overlay"), Mapping)]
        frontier = pareto_front(
            [
                _parse_objectives(row.get("objectives") if isinstance(row.get("objectives"), Mapping) else {})
                for row in history
                if isinstance(row, Mapping)
            ]
        )

        best = None
        best_acq = -10_000.0
        for _ in range(candidate_pool):
            candidate_overlay = self._random_candidate(current_overlay)
            candidate_flat = _flatten_overlay(candidate_overlay)
            mean, uncertainty, objectives = self._surrogate(candidate_flat, history)
            ei = mean - best_score_hist
            acq = mean + (kappa * uncertainty) + ei

            if seen_flats:
                nearest = min(
                    _distance(candidate_flat, flat, self.param_specs)
                    for flat in seen_flats
                    if isinstance(flat, Mapping)
                )
                acq += 0.08 * nearest

            if frontier:
                dominated = any(_dominates(front_obj, objectives) for front_obj in frontier if front_obj)
                if not dominated:
                    acq += 0.12

            if acq > best_acq:
                best_acq = acq
                best = {
                    "overlay": candidate_overlay,
                    "acquisition": round(acq, 6),
                    "expected_score": round(mean, 6),
                    "uncertainty": round(uncertainty, 6),
                    "predicted_objectives": objectives,
                }

        self.state["attempts"] = int(self.state.get("attempts") or 0) + 1
        return best or {"overlay": dict(current_overlay)}

    def observe_result(
        self,
        *,
        accepted: bool,
        score: float | None = None,
        overlay: Mapping[str, Any] | None = None,
        objectives: Mapping[str, float] | None = None,
        report: Mapping[str, Any] | None = None,
    ) -> None:
        if accepted:
            self.state["accepted"] = int(self.state.get("accepted") or 0) + 1
        history = self._history()
        entry = {
            "score": _safe_float(score, 0.0),
            "accepted": bool(accepted),
            "flat_overlay": _flatten_overlay(dict(overlay or {})),
            "objectives": dict(_parse_objectives(objectives)),
            "trial_id": str((report or {}).get("trial_id") or ""),
        }
        history.append(entry)
        self.state["history"] = history[-200:]

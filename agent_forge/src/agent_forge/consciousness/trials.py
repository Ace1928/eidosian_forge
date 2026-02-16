from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from agent_forge.core import db as DB
from agent_forge.core import events
from agent_forge.core import workspace

from .kernel import ConsciousnessKernel
from .metrics import coherence_from_workspace_summary, response_complexity
from .perturb import Perturbation, apply_perturbation


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _forge_root() -> Path:
    return Path(os.environ.get("EIDOS_FORGE_DIR", Path(__file__).resolve().parents[4])).resolve()


def _trial_report_dir() -> Path:
    default = _forge_root() / "reports" / "consciousness_trials"
    path = Path(os.environ.get("EIDOS_CONSCIOUSNESS_TRIAL_DIR", str(default))).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _latest_metric(items: list[dict[str, Any]], key: str) -> Optional[float]:
    for evt in reversed(items):
        if str(evt.get("type") or "") != "metrics.sample":
            continue
        data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
        if str(data.get("key") or "") != key:
            continue
        try:
            return float(data.get("value"))
        except (TypeError, ValueError):
            return None
    return None


def _metrics_snapshot(state_dir: Path, recent_events: list[dict[str, Any]]) -> dict[str, Any]:
    ws = workspace.summary(state_dir, limit=400, window_seconds=1.0, min_sources=3)
    coherence = coherence_from_workspace_summary(ws)
    rci = response_complexity(recent_events[-250:])
    return {
        "workspace": ws,
        "coherence": coherence,
        "rci": rci,
        "agency": _latest_metric(recent_events, "consciousness.agency"),
        "boundary_stability": _latest_metric(recent_events, "consciousness.boundary_stability"),
    }


def _numeric_delta(after: Any, before: Any) -> Optional[float]:
    if not isinstance(after, (int, float)) or not isinstance(before, (int, float)):
        return None
    return round(float(after) - float(before), 6)


def _apply_overrides(kernel: ConsciousnessKernel, perturbation: Perturbation) -> dict[str, Any]:
    cfg = kernel.config
    old = dict(cfg)
    kind = perturbation.kind
    target = perturbation.target
    magnitude = float(perturbation.magnitude)

    if kind == "noise" and target in {"attention", "*"}:
        cfg["attention_score_noise"] = max(0.0, min(1.0, magnitude))
    elif kind == "drop" and target in {"workspace_competition", "gw", "*"}:
        cfg["competition_drop_winners"] = True
    elif kind == "zero" and target in {"policy", "*"}:
        cfg["policy_max_actions_per_tick"] = 0
    elif kind == "jitter" and target in {"competition", "workspace_competition", "*"}:
        cfg["competition_reaction_window_secs"] = float(cfg.get("competition_reaction_window_secs", 1.5)) + max(0.0, magnitude)

    return old


@dataclass
class TrialResult:
    report_id: str
    report_path: Optional[Path]
    report: Dict[str, Any]


class ConsciousnessTrialRunner:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)

    def run_trial(
        self,
        *,
        kernel: Optional[ConsciousnessKernel] = None,
        perturbation: Perturbation,
        ticks: int = 3,
        persist: bool = True,
    ) -> TrialResult:
        kernel = kernel or ConsciousnessKernel(self.state_dir)

        before_events = events.iter_events(self.state_dir, limit=600)
        before = _metrics_snapshot(self.state_dir, before_events)

        inject_payload = {
            "id": uuid.uuid4().hex,
            "kind": perturbation.kind,
            "target": perturbation.target,
            "magnitude": float(perturbation.magnitude),
            "duration_s": float(perturbation.duration_s),
            "meta": dict(perturbation.meta),
            "ts": _now_iso(),
        }
        inject_evt = events.append(
            self.state_dir,
            "perturb.inject",
            inject_payload,
            tags=["consciousness", "perturb"],
        )

        harness_result = apply_perturbation(kernel, perturbation)
        original = _apply_overrides(kernel, perturbation)
        try:
            for _ in range(max(1, int(ticks))):
                kernel.tick()
        finally:
            kernel.config.clear()
            kernel.config.update(original)

        after_events = events.iter_events(self.state_dir, limit=800)
        after = _metrics_snapshot(self.state_dir, after_events)

        delta = {
            "ignition_delta": _numeric_delta(
                (after.get("workspace") or {}).get("ignition_count"),
                (before.get("workspace") or {}).get("ignition_count"),
            ),
            "coherence_delta": _numeric_delta(
                (after.get("coherence") or {}).get("coherence_ratio"),
                (before.get("coherence") or {}).get("coherence_ratio"),
            ),
            "rci_delta": _numeric_delta(
                ((after.get("rci") or {}).get("rci")),
                ((before.get("rci") or {}).get("rci")),
            ),
            "agency_delta": _numeric_delta(after.get("agency"), before.get("agency")),
            "boundary_delta": _numeric_delta(after.get("boundary_stability"), before.get("boundary_stability")),
        }

        report_id = f"trial_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
        report = {
            "report_id": report_id,
            "timestamp": _now_iso(),
            "state_dir": str(self.state_dir),
            "ticks": int(ticks),
            "perturbation": inject_payload,
            "harness_result": {
                "applied": harness_result.applied,
                "details": harness_result.details,
            },
            "before": before,
            "after": after,
            "delta": delta,
        }

        events.append(
            self.state_dir,
            "perturb.response",
            {
                "report_id": report_id,
                "inject_event_corr_id": inject_evt.get("corr_id"),
                "delta": delta,
            },
            tags=["consciousness", "perturb", "response"],
            corr_id=inject_evt.get("corr_id"),
            parent_id=inject_evt.get("parent_id"),
        )

        rci_value = (after.get("rci") or {}).get("rci")
        if isinstance(rci_value, (int, float)):
            DB.insert_metric(self.state_dir, "consciousness.rci", float(rci_value))
            events.append(
                self.state_dir,
                "metrics.sample",
                {"key": "consciousness.rci", "value": float(rci_value), "ts": _now_iso()},
                tags=["metrics", "consciousness"],
            )

        report_path: Optional[Path] = None
        if persist:
            report_path = _trial_report_dir() / f"{report_id}.json"
            report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            report["report_path"] = str(report_path)

        return TrialResult(report_id=report_id, report_path=report_path, report=report)

    def latest_trial(self) -> Optional[dict[str, Any]]:
        files = sorted(_trial_report_dir().glob("trial_*.json"))
        if not files:
            return None
        latest = max(files, key=lambda p: p.stat().st_mtime_ns)
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            return None

    def status(self) -> dict[str, Any]:
        recent = events.iter_events(self.state_dir, limit=500)
        ws = workspace.summary(self.state_dir, limit=400, window_seconds=1.0, min_sources=3)
        coherence = coherence_from_workspace_summary(ws)
        rci = response_complexity(recent[-250:])
        return {
            "timestamp": _now_iso(),
            "state_dir": str(self.state_dir),
            "workspace": ws,
            "coherence": coherence,
            "rci": rci,
            "agency": _latest_metric(recent, "consciousness.agency"),
            "boundary_stability": _latest_metric(recent, "consciousness.boundary_stability"),
            "latest_trial": self.latest_trial(),
        }

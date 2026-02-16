from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from agent_forge.core import events, workspace

from .kernel import ConsciousnessKernel
from .metrics import coherence_from_workspace_summary, response_complexity

KNOWN_EXTERNAL_BENCHMARKS: dict[str, dict[str, Any]] = {
    "mmlu": {"target": 0.7, "higher_is_better": True},
    "gpqa": {"target": 0.5, "higher_is_better": True},
    "swe_bench_verified": {"target": 0.35, "higher_is_better": True},
    "human_eval": {"target": 0.75, "higher_is_better": True},
}


def _forge_root() -> Path:
    return Path(os.environ.get("EIDOS_FORGE_DIR", Path(__file__).resolve().parents[4])).resolve()


def _benchmark_dir() -> Path:
    default = _forge_root() / "reports" / "consciousness_benchmarks"
    path = Path(os.environ.get("EIDOS_CONSCIOUSNESS_BENCHMARK_DIR", str(default))).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def _latest_event_data(
    items: list[dict[str, Any]],
    etype: str,
) -> Optional[dict[str, Any]]:
    for evt in reversed(items):
        if str(evt.get("type") or "") != etype:
            continue
        data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
        return dict(data)
    return None


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round((len(vals) - 1) * max(0.0, min(1.0, p))))
    return float(vals[idx])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalize_external_scores(
    external_scores: Mapping[str, float],
    external_sources: Mapping[str, str],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    norm_values: list[float] = []

    for name, value in external_scores.items():
        score = _safe_float(value, default=-1.0)
        spec = KNOWN_EXTERNAL_BENCHMARKS.get(name.lower())
        if spec:
            target = float(spec["target"])
            hib = bool(spec["higher_is_better"])
            if hib:
                norm = score / target if target > 0 else 0.0
            else:
                norm = target / max(score, 1e-9)
        else:
            norm = score
        norm = max(0.0, min(2.0, norm))
        norm_values.append(norm)
        normalized[name] = {
            "score": score,
            "normalized": round(norm, 6),
            "source": str(external_sources.get(name) or ""),
            "spec": spec or {},
        }

    return {
        "scores": normalized,
        "external_index": round(sum(norm_values) / max(len(norm_values), 1), 6) if norm_values else None,
    }


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@dataclass
class BenchmarkResult:
    benchmark_id: str
    report_path: Optional[Path]
    report: Dict[str, Any]


class ConsciousnessBenchmarkSuite:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)

    def latest_benchmark(self) -> Optional[dict[str, Any]]:
        files = sorted(_benchmark_dir().glob("benchmark_*.json"))
        if not files:
            return None
        latest = max(files, key=lambda path: path.stat().st_mtime_ns)
        return _load_json(latest)

    def run(
        self,
        *,
        kernel: Optional[ConsciousnessKernel] = None,
        ticks: int = 12,
        persist: bool = True,
        external_scores: Optional[Mapping[str, float]] = None,
        external_sources: Optional[Mapping[str, str]] = None,
        baseline_report: Optional[str] = None,
    ) -> BenchmarkResult:
        kernel = kernel or ConsciousnessKernel(self.state_dir)
        ticks = max(1, int(ticks))

        lat_ms: list[float] = []
        emitted_total = 0
        started = time.perf_counter()
        for _ in range(ticks):
            t0 = time.perf_counter()
            res = kernel.tick()
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
            emitted_total += int(res.emitted_events)
        wall_ms = (time.perf_counter() - started) * 1000.0

        recent = events.iter_events(self.state_dir, limit=900)
        ws = workspace.summary(self.state_dir, limit=400, window_seconds=1.0, min_sources=3)
        coherence = coherence_from_workspace_summary(ws)
        rci = response_complexity(recent[-300:])
        memory_status = _latest_event_data(recent, "memory_bridge.status") or {}
        knowledge_status = _latest_event_data(recent, "knowledge_bridge.status") or {}

        perf = {
            "ticks": ticks,
            "tick_latency_ms_p50": round(_percentile(lat_ms, 0.5), 6),
            "tick_latency_ms_p95": round(_percentile(lat_ms, 0.95), 6),
            "tick_latency_ms_max": round(max(lat_ms) if lat_ms else 0.0, 6),
            "events_emitted_total": emitted_total,
            "events_emitted_per_tick": round(emitted_total / max(ticks, 1), 6),
            "wall_time_ms": round(wall_ms, 6),
        }

        capability = {
            "coherence_ratio": round(float(coherence.get("coherence_ratio") or 0.0), 6),
            "rci": round(float(rci.get("rci") or 0.0), 6),
            "agency": _latest_metric(recent, "consciousness.agency"),
            "boundary_stability": _latest_metric(recent, "consciousness.boundary_stability"),
            "prediction_error": _latest_metric(recent, "consciousness.world.prediction_error"),
            "meta_confidence": _latest_metric(recent, "consciousness.meta.confidence"),
            "report_groundedness": _latest_metric(recent, "consciousness.report.groundedness"),
            "memory_recalls": _latest_metric(recent, "consciousness.memory_bridge.recalls"),
            "knowledge_hits": _latest_metric(recent, "consciousness.knowledge_bridge.total_hits"),
            "memory_bridge_available": memory_status.get("available"),
            "knowledge_bridge_available": knowledge_status.get("available"),
        }

        pred_err = _safe_float(capability.get("prediction_error"), default=1.0)
        memory_recall_norm = _clamp(
            _safe_float(capability.get("memory_recalls"), default=0.0)
            / max(float(kernel.config.get("memory_bridge_recall_limit", 4)), 1.0),
            0.0,
            1.0,
        )
        knowledge_hits_norm = _clamp(
            _safe_float(capability.get("knowledge_hits"), default=0.0)
            / max(float(kernel.config.get("knowledge_bridge_context_limit", 6)), 1.0),
            0.0,
            1.0,
        )
        cap_index = (
            0.22 * _safe_float(capability.get("coherence_ratio"))
            + 0.18 * _safe_float(capability.get("rci"))
            + 0.12 * _safe_float(capability.get("agency"))
            + 0.08 * _safe_float(capability.get("boundary_stability"))
            + 0.13 * (1.0 - min(1.0, pred_err))
            + 0.12 * _safe_float(capability.get("report_groundedness"))
            + 0.08 * memory_recall_norm
            + 0.07 * knowledge_hits_norm
        )
        perf_index = 1.0 / max(1.0, perf["tick_latency_ms_p95"] / 10.0)

        external = _normalize_external_scores(
            external_scores or {},
            external_sources or {},
        )
        external_index = _safe_float(external.get("external_index"), default=0.0)
        composite = round(0.55 * cap_index + 0.35 * perf_index + 0.10 * external_index, 6)

        baseline: Optional[dict[str, Any]] = None
        if baseline_report:
            baseline = _load_json(Path(baseline_report).expanduser().resolve())
        if baseline is None:
            baseline = self.latest_benchmark()

        baseline_composite = None
        if isinstance(baseline, Mapping):
            baseline_composite = _safe_float((baseline.get("scores") or {}).get("composite"), default=0.0)
        delta_composite = None
        improved = None
        if baseline_composite is not None:
            delta_composite = round(composite - baseline_composite, 6)
            improved = bool(delta_composite >= -0.01)

        gates = {
            "world_model_online": capability.get("prediction_error") is not None,
            "meta_online": capability.get("meta_confidence") is not None,
            "report_online": capability.get("report_groundedness") is not None,
            "memory_bridge_observed": capability.get("memory_bridge_available") is not None,
            "knowledge_bridge_observed": capability.get("knowledge_bridge_available") is not None,
            "latency_p95_under_100ms": bool(perf["tick_latency_ms_p95"] < 100.0),
            "non_regression_vs_baseline": improved if improved is not None else True,
        }

        benchmark_id = f"benchmark_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
        report = {
            "benchmark_id": benchmark_id,
            "timestamp": _now_iso(),
            "state_dir": str(self.state_dir),
            "performance": perf,
            "capability": capability,
            "workspace": ws,
            "coherence": coherence,
            "rci": rci,
            "external": external,
            "scores": {
                "capability_index": round(cap_index, 6),
                "performance_index": round(perf_index, 6),
                "external_index": round(external_index, 6) if external.get("external_index") is not None else None,
                "composite": composite,
                "baseline_composite": baseline_composite,
                "delta_composite": delta_composite,
            },
            "gates": gates,
        }

        events.append(
            self.state_dir,
            "benchmark.run",
            {
                "benchmark_id": benchmark_id,
                "scores": report["scores"],
                "gates": gates,
                "performance": {
                    "tick_latency_ms_p95": perf["tick_latency_ms_p95"],
                    "events_emitted_per_tick": perf["events_emitted_per_tick"],
                },
            },
            tags=["consciousness", "benchmark"],
        )

        path: Optional[Path] = None
        if persist:
            path = _benchmark_dir() / f"{benchmark_id}.json"
            path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            report["report_path"] = str(path)
        return BenchmarkResult(benchmark_id=benchmark_id, report_path=path, report=report)

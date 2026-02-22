from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: Any) -> Optional[float]:
    parsed = _safe_float(value, None)
    if parsed is None:
        return None
    return max(0.0, min(1.0, float(parsed)))


def _flatten_metrics(payload: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_metrics(value, path))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            path = f"{prefix}[{idx}]"
            out.update(_flatten_metrics(value, path))
    else:
        parsed = _safe_float(payload, None)
        if parsed is not None and prefix:
            out[prefix] = float(parsed)
    return out


@dataclass
class ImportedExternalBenchmark:
    benchmark_id: str
    report_path: Optional[Path]
    report: dict[str, Any]


class ExternalBenchmarkImporter:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)

    def _benchmark_dir(self) -> Path:
        default = Path(os.environ.get("EIDOS_FORGE_DIR", Path(__file__).resolve().parents[4])).resolve()
        root = default / "reports" / "consciousness_benchmarks"
        path = Path(os.environ.get("EIDOS_CONSCIOUSNESS_BENCHMARK_DIR", str(root))).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _pick_metric(self, metrics: Mapping[str, float], keys: list[str]) -> Optional[float]:
        for key in keys:
            if key in metrics:
                return metrics[key]
        return None

    def import_file(
        self,
        *,
        path: str | Path,
        suite: str,
        name: Optional[str] = None,
        source_url: Optional[str] = None,
        persist: bool = True,
    ) -> ImportedExternalBenchmark:
        source_path = Path(path).expanduser().resolve()
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        metrics = _flatten_metrics(payload)
        suite_key = suite.strip().lower().replace("_", "-")

        score_candidates = {
            "swe-bench": ["resolved_rate", "pass@1", "score", "metrics.resolved_rate", "metrics.pass@1"],
            "webarena": ["success_rate", "score", "task_success_rate", "metrics.success_rate"],
            "osworld": ["success_rate", "score", "task_success_rate", "metrics.success_rate"],
            "agentbench": ["success_rate", "score", "metrics.success_rate", "overall"],
            "generic": ["score", "success_rate", "metrics.score", "metrics.success_rate"],
        }
        reliability_candidates = {
            "swe-bench": ["stability", "consistency", "metrics.consistency"],
            "webarena": ["consistency", "reliability"],
            "osworld": ["consistency", "reliability"],
            "agentbench": ["consistency", "reliability"],
            "generic": ["reliability", "consistency"],
        }
        task_candidates = {
            "swe-bench": ["resolved_rate", "pass@1"],
            "webarena": ["task_success_rate", "success_rate"],
            "osworld": ["task_success_rate", "success_rate"],
            "agentbench": ["task_success_rate", "success_rate"],
            "generic": ["success_rate", "score"],
        }

        score_raw = self._pick_metric(metrics, score_candidates.get(suite_key, score_candidates["generic"]))
        task_raw = self._pick_metric(metrics, task_candidates.get(suite_key, task_candidates["generic"]))
        reliability_raw = self._pick_metric(
            metrics, reliability_candidates.get(suite_key, reliability_candidates["generic"])
        )
        score = _clamp01(score_raw)
        task_score = _clamp01(task_raw if task_raw is not None else score_raw)
        reliability = _clamp01(reliability_raw)

        benchmark_id = (
            f"benchmark_external_{suite_key}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
        )
        report = {
            "benchmark_id": benchmark_id,
            "timestamp": _now_iso(),
            "state_dir": str(self.state_dir),
            "source": "external_import",
            "suite": suite_key,
            "name": name or source_path.stem,
            "capability": {
                "coherence_ratio": score,
                "trace_strength": score,
                "agency": task_score,
                "boundary_stability": reliability,
                "prediction_error": None,
                "meta_confidence": reliability,
                "report_groundedness": score,
                "phenom_unity_index": None,
                "phenom_continuity_index": None,
                "phenom_ownership_index": None,
                "phenom_perspective_coherence_index": None,
                "phenom_dream_likeness_index": None,
            },
            "scores": {
                "composite": score,
            },
            "external": {
                "source_url": source_url,
                "input_path": str(source_path),
                "metric_count": len(metrics),
                "raw_metrics": metrics,
            },
        }

        report_path: Optional[Path] = None
        if persist:
            report_path = self._benchmark_dir() / f"{benchmark_id}.json"
            report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            report["report_path"] = str(report_path)

        return ImportedExternalBenchmark(benchmark_id=benchmark_id, report_path=report_path, report=report)

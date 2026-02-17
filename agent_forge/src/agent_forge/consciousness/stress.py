from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from agent_forge.core import events

from .kernel import ConsciousnessKernel
from .types import TickContext


def _forge_root() -> Path:
    return Path(os.environ.get("EIDOS_FORGE_DIR", Path(__file__).resolve().parents[4])).resolve()


def _stress_dir() -> Path:
    default = _forge_root() / "reports" / "consciousness_stress_benchmarks"
    path = Path(os.environ.get("EIDOS_CONSCIOUSNESS_STRESS_BENCHMARK_DIR", str(default))).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, Mapping):
        return dict(payload)
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round((len(vals) - 1) * max(0.0, min(1.0, p))))
    return float(vals[idx])


def _event_bus_bytes(state_dir: Path) -> int:
    events_root = state_dir / "events"
    if not events_root.exists():
        return 0
    return sum(path.stat().st_size for path in events_root.rglob("bus-*.jsonl"))


def _iter_recent(state_dir: Path, limit: int) -> list[dict[str, Any]]:
    return events.iter_events(state_dir, limit=max(0, int(limit)))


def _count_window(
    items: Iterable[dict[str, Any]],
    *,
    since_ts: str,
    etype: str = "",
    source: str = "",
) -> int:
    count = 0
    for evt in items:
        ts = str(evt.get("ts") or "")
        if ts and ts < since_ts:
            continue
        if etype and str(evt.get("type") or "") != etype:
            continue
        if source:
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            if str(data.get("source") or "") != source:
                continue
        count += 1
    return count


class _StressEmitterModule:
    name = "stress_emitter"

    def __init__(self, *, event_fanout: int, broadcast_fanout: int, payload_chars: int) -> None:
        self.event_fanout = max(1, int(event_fanout))
        self.broadcast_fanout = max(0, int(broadcast_fanout))
        self.payload_chars = max(32, int(payload_chars))

    def tick(self, ctx: TickContext) -> None:
        blob = "x" * self.payload_chars
        for idx in range(self.event_fanout):
            ctx.emit_event(
                "stress.synthetic",
                {
                    "source_module": self.name,
                    "beat": int(ctx.beat_count),
                    "index": int(idx),
                    "content": {
                        "blob": blob,
                        "nested": {"blob_copy": blob, "beat": int(ctx.beat_count)},
                    },
                },
                tags=["consciousness", "stress"],
            )
        for idx in range(self.broadcast_fanout):
            ctx.broadcast(
                self.name,
                {
                    "kind": "METRIC",
                    "id": f"stress-{ctx.beat_count}-{idx}",
                    "source_module": self.name,
                    "confidence": 0.45,
                    "salience": 0.85,
                    "content": {
                        "stress": True,
                        "beat": int(ctx.beat_count),
                        "index": int(idx),
                        "blob": blob,
                    },
                },
                channel="stress",
            )


@dataclass
class StressBenchmarkResult:
    benchmark_id: str
    report_path: Optional[Path]
    report: Dict[str, Any]


class ConsciousnessStressBenchmark:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)

    def latest_stress_benchmark(self) -> Optional[dict[str, Any]]:
        files = sorted(_stress_dir().glob("stress_*.json"))
        if not files:
            return None
        latest = max(files, key=lambda path: path.stat().st_mtime_ns)
        return _load_json(latest)

    def run(
        self,
        *,
        ticks: int = 20,
        event_fanout: int = 14,
        broadcast_fanout: int = 6,
        payload_chars: int = 12_000,
        max_payload_bytes: int = 2_048,
        persist: bool = True,
    ) -> StressBenchmarkResult:
        ticks = max(1, int(ticks))
        event_fanout = max(1, int(event_fanout))
        broadcast_fanout = max(0, int(broadcast_fanout))
        payload_chars = max(64, int(payload_chars))
        max_payload_bytes = max(512, int(max_payload_bytes))

        kernel = ConsciousnessKernel(
            self.state_dir,
            modules=[
                _StressEmitterModule(
                    event_fanout=event_fanout,
                    broadcast_fanout=broadcast_fanout,
                    payload_chars=payload_chars,
                )
            ],
            config={
                "kernel_watchdog_enabled": True,
                "consciousness_max_payload_bytes": max_payload_bytes,
                "consciousness_payload_truncation_event": True,
                "recent_events_limit": max(400, ticks * (event_fanout + broadcast_fanout) * 2),
                "recent_broadcast_limit": max(200, ticks * (broadcast_fanout + 4)),
            },
        )

        start_ts = _now_iso()
        bus_bytes_before = _event_bus_bytes(self.state_dir)
        lat_ms: list[float] = []
        emitted_total = 0

        started = time.perf_counter()
        for _ in range(ticks):
            t0 = time.perf_counter()
            result = kernel.tick()
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
            emitted_total += int(result.emitted_events)
        wall_s = max(time.perf_counter() - started, 1e-9)
        bus_bytes_after = _event_bus_bytes(self.state_dir)

        sample_limit = max(2000, ticks * (event_fanout + broadcast_fanout) * 6)
        recent = _iter_recent(self.state_dir, limit=sample_limit)

        trunc_count = _count_window(
            recent,
            since_ts=start_ts,
            etype="consciousness.payload_truncated",
        )
        synthetic_events = _count_window(
            recent,
            since_ts=start_ts,
            etype="stress.synthetic",
        )
        stress_broadcasts = _count_window(
            recent,
            since_ts=start_ts,
            etype="workspace.broadcast",
            source="stress_emitter",
        )
        module_errors = _count_window(
            recent,
            since_ts=start_ts,
            etype="consciousness.module_error",
        )
        watchdog = kernel.watchdog_status()

        perf = {
            "ticks": ticks,
            "tick_latency_ms_p50": round(_percentile(lat_ms, 0.5), 6),
            "tick_latency_ms_p95": round(_percentile(lat_ms, 0.95), 6),
            "tick_latency_ms_max": round(max(lat_ms) if lat_ms else 0.0, 6),
            "events_emitted_total": int(emitted_total),
            "events_emitted_per_tick": round(float(emitted_total) / max(float(ticks), 1.0), 6),
            "events_per_second": round(float(emitted_total) / wall_s, 6),
            "wall_time_seconds": round(wall_s, 6),
            "event_bus_growth_bytes": int(max(0, bus_bytes_after - bus_bytes_before)),
        }
        pressure = {
            "synthetic_events_observed": int(synthetic_events),
            "stress_broadcasts_observed": int(stress_broadcasts),
            "payload_truncations_observed": int(trunc_count),
            "truncation_rate_per_emitted_event": round(
                float(trunc_count) / max(float(emitted_total), 1.0),
                6,
            ),
            "module_error_count": int(module_errors),
        }
        gates = {
            "payload_truncation_observed": bool(trunc_count > 0),
            "event_pressure_hits_target": bool(
                _safe_float(perf.get("events_emitted_per_tick")) >= float(event_fanout + broadcast_fanout)
            ),
            "latency_p95_under_200ms": bool(_safe_float(perf.get("tick_latency_ms_p95")) < 200.0),
            "module_error_free": bool(module_errors == 0),
            "watchdog_no_quarantine": bool(
                int(watchdog.get("quarantined_modules") or 0) == 0
            ),
        }

        benchmark_id = f"stress_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
        report = {
            "benchmark_id": benchmark_id,
            "timestamp": _now_iso(),
            "state_dir": str(self.state_dir),
            "profile": {
                "ticks": ticks,
                "event_fanout": event_fanout,
                "broadcast_fanout": broadcast_fanout,
                "payload_chars": payload_chars,
                "max_payload_bytes": max_payload_bytes,
            },
            "performance": perf,
            "pressure": pressure,
            "watchdog": watchdog,
            "gates": gates,
        }

        events.append(
            self.state_dir,
            "benchmark.stress_run",
            {
                "benchmark_id": benchmark_id,
                "profile": report["profile"],
                "performance": {
                    "tick_latency_ms_p95": perf["tick_latency_ms_p95"],
                    "events_per_second": perf["events_per_second"],
                },
                "pressure": {
                    "truncations": pressure["payload_truncations_observed"],
                    "module_errors": pressure["module_error_count"],
                },
                "gates": gates,
            },
            tags=["consciousness", "benchmark", "stress"],
        )

        path: Optional[Path] = None
        if persist:
            path = _stress_dir() / f"{benchmark_id}.json"
            path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            report["report_path"] = str(path)
        return StressBenchmarkResult(
            benchmark_id=benchmark_id,
            report_path=path,
            report=report,
        )

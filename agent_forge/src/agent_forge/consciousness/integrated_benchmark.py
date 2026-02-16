from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from agent_forge.core import events

from .bench.red_team import ConsciousnessRedTeamCampaign
from .benchmarks import ConsciousnessBenchmarkSuite
from .kernel import ConsciousnessKernel
from .perturb import make_drop, make_noise
from .trials import ConsciousnessTrialRunner


def _forge_root() -> Path:
    return Path(os.environ.get("EIDOS_FORGE_DIR", Path(__file__).resolve().parents[4])).resolve()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _report_dir() -> Path:
    default = _forge_root() / "reports" / "consciousness_integrated_benchmarks"
    path = Path(os.environ.get("EIDOS_CONSCIOUSNESS_INTEGRATED_BENCH_DIR", str(default))).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _weighted_score(components: list[tuple[float, float, bool]]) -> float:
    active_weight = sum(weight for _, weight, enabled in components if enabled and weight > 0.0)
    if active_weight <= 0.0:
        return 0.0
    total = sum(score * weight for score, weight, enabled in components if enabled and weight > 0.0)
    return round(float(total) / float(active_weight), 6)


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _post_json(url: str, payload: dict[str, Any], timeout: float = 45.0) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        parsed = json.loads(body)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return parsed, latency_ms


def _get_json(url: str, timeout: float = 15.0) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        parsed = json.loads(body)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return parsed, latency_ms


def _llm_tasks() -> list[dict[str, Any]]:
    return [
        {
            "id": "arith_sum",
            "prompt": "Return only JSON: {\"sum\": 611+223}",
            "validator": lambda response: _safe_float(_extract_json_field(response, "sum"), default=-1) == 834.0,
        },
        {
            "id": "prime_select",
            "prompt": "Return only JSON: {\"prime\": <prime number from [21, 29, 35]>}",
            "validator": lambda response: int(_safe_float(_extract_json_field(response, "prime"), default=-1)) == 29,
        },
        {
            "id": "state_reasoning",
            "prompt": (
                "Return only JSON: "
                "{\"mode\":\"grounded|simulated|degraded\",\"reason\":\"<=8 words\"}. "
                "Choose mode=grounded if coherence high and prediction error low."
            ),
            "validator": lambda response: str(_extract_json_field(response, "mode")).lower()
            in {"grounded", "simulated", "degraded"},
        },
        {
            "id": "idempotence",
            "prompt": "Return only JSON: {\"idempotent\": true}",
            "validator": lambda response: bool(_extract_json_field(response, "idempotent")) is True,
        },
        {
            "id": "rollback",
            "prompt": "Return only JSON: {\"rollback_steps\": 3}",
            "validator": lambda response: int(_safe_float(_extract_json_field(response, "rollback_steps"), default=-1)) == 3,
        },
    ]


def _extract_json_field(response: str, key: str) -> Any:
    try:
        data = json.loads(response)
        if isinstance(data, Mapping):
            return data.get(key)
    except Exception:
        pass
    return None


async def _run_mcp_suite(state_dir: Path, timeout_sec: float = 45.0) -> dict[str, Any]:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except Exception as exc:
        return {"available": False, "error": f"mcp-client-unavailable: {exc}"}

    root = _forge_root()
    python_bin = str((root / "eidosian_venv" / "bin" / "python3"))
    if not Path(python_bin).exists():
        python_bin = sys.executable
    env = {
        **os.environ,
        "PYTHONPATH": f"{root}/eidos_mcp/src:{root}/lib:{root}/agent_forge/src:{root}",
        "EIDOS_FORGE_DIR": str(root),
        "EIDOS_MCP_TRANSPORT": "stdio",
        "EIDOS_MCP_STATELESS_HTTP": "1",
    }
    params = StdioServerParameters(
        command=python_bin,
        args=["-u", "-c", "import eidos_mcp.eidos_mcp_server as s; s.main()"],
        env=env,
    )

    cases: list[tuple[str, dict[str, Any], Callable[[str], bool]]] = [
        ("diagnostics_ping", {}, lambda r: r.strip() == "ok"),
        ("system_info", {}, lambda r: "Linux" in r or "Android" in r),
        ("consciousness_kernel_status", {"state_dir": str(state_dir)}, lambda r: "\"workspace\"" in r and "\"rci\"" in r),
        (
            "consciousness_bridge_status",
            {"state_dir": str(state_dir)},
            lambda r: "\"memory_bridge\"" in r and "\"knowledge_bridge\"" in r,
        ),
        (
            "consciousness_kernel_benchmark",
            {"state_dir": str(state_dir), "ticks": 2, "persist": False},
            lambda r: "\"benchmark_id\"" in r and "\"composite\"" in r,
        ),
    ]

    outcomes: list[dict[str, Any]] = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=timeout_sec)
            for name, arguments, validator in cases:
                t0 = time.perf_counter()
                ok = False
                error = ""
                text = ""
                try:
                    result = await asyncio.wait_for(session.call_tool(name, arguments=arguments), timeout=timeout_sec)
                    if result.structuredContent and "result" in result.structuredContent:
                        text = str(result.structuredContent["result"])
                    elif result.content:
                        for content in result.content:
                            if getattr(content, "type", None) == "text":
                                text = content.text
                                break
                    ok = bool(validator(text))
                except Exception as exc:
                    error = str(exc)
                    ok = False
                outcomes.append(
                    {
                        "tool": name,
                        "ok": ok,
                        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 6),
                        "error": error,
                        "sample": text[:240],
                    }
                )

    success_rate = sum(1 for row in outcomes if row["ok"]) / max(len(outcomes), 1)
    latency_p95 = sorted([row["latency_ms"] for row in outcomes])[-1] if outcomes else 0.0
    return {
        "available": True,
        "outcomes": outcomes,
        "success_rate": round(float(success_rate), 6),
        "latency_ms_p95": round(float(latency_p95), 6),
    }


@dataclass
class IntegratedBenchmarkResult:
    benchmark_id: str
    report_path: Optional[Path]
    report: Dict[str, Any]


class IntegratedStackBenchmark:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)

    def latest(self) -> Optional[dict[str, Any]]:
        files = sorted(_report_dir().glob("integrated_*.json"))
        if not files:
            return None
        latest = max(files, key=lambda p: p.stat().st_mtime_ns)
        return _load_json(latest)

    def _run_local_llm(self, *, model: str, endpoint: str, timeout_sec: float) -> dict[str, Any]:
        tasks = _llm_tasks()
        outcomes: list[dict[str, Any]] = []
        tags_latency = None
        available = True
        error = ""
        try:
            _, tags_latency = _get_json(f"{endpoint}/api/tags", timeout=timeout_sec)
        except Exception as exc:
            available = False
            error = str(exc)

        if not available:
            return {"available": False, "error": error, "outcomes": [], "success_rate": 0.0, "latency_ms_p95": None}

        for task in tasks:
            t0 = time.perf_counter()
            ok = False
            resp = ""
            err = ""
            try:
                payload = {
                    "model": model,
                    "prompt": task["prompt"],
                    "stream": False,
                }
                out, _ = _post_json(f"{endpoint}/api/generate", payload=payload, timeout=timeout_sec)
                resp = str(out.get("response") or "").strip()
                ok = bool(task["validator"](resp))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                err = str(exc)
                ok = False
            outcomes.append(
                {
                    "id": task["id"],
                    "ok": ok,
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 6),
                    "error": err,
                    "sample": resp[:200],
                }
            )

        success_rate = sum(1 for row in outcomes if row["ok"]) / max(len(outcomes), 1)
        latency_vals = sorted([row["latency_ms"] for row in outcomes]) if outcomes else [0.0]
        p95 = latency_vals[int((len(latency_vals) - 1) * 0.95)] if latency_vals else 0.0
        return {
            "available": True,
            "model": model,
            "tags_latency_ms": round(float(tags_latency or 0.0), 6),
            "outcomes": outcomes,
            "success_rate": round(float(success_rate), 6),
            "latency_ms_p95": round(float(p95), 6),
        }

    def run(
        self,
        *,
        rounds: int = 3,
        bench_ticks: int = 10,
        trial_ticks: int = 3,
        run_mcp: bool = True,
        run_llm: bool = True,
        run_red_team: bool = False,
        red_team_quick: bool = True,
        red_team_max_scenarios: int = 1,
        red_team_seed: int = 910_000,
        persist: bool = True,
        llm_model: str = "qwen2.5:1.5b",
        ollama_endpoint: str = "http://127.0.0.1:11434",
        timeout_sec: float = 45.0,
    ) -> IntegratedBenchmarkResult:
        rounds = max(1, int(rounds))
        bench_ticks = max(1, int(bench_ticks))
        trial_ticks = max(1, int(trial_ticks))

        bench_suite = ConsciousnessBenchmarkSuite(self.state_dir)
        trial_runner = ConsciousnessTrialRunner(self.state_dir)

        core_runs: list[dict[str, Any]] = []
        for _ in range(rounds):
            kernel = ConsciousnessKernel(self.state_dir)
            out = bench_suite.run(kernel=kernel, ticks=bench_ticks, persist=False)
            core_runs.append(out.report)

        trial_specs = [
            make_noise("attention", 0.25, 1.0),
            make_drop("workspace_competition", 1.0),
            make_noise("attention", 0.4, 1.0),
        ]
        trials: list[dict[str, Any]] = []
        for perturbation in trial_specs:
            kernel = ConsciousnessKernel(self.state_dir)
            tr = trial_runner.run_trial(
                kernel=kernel,
                perturbation=perturbation,
                ticks=trial_ticks,
                persist=False,
            )
            trials.append(tr.report)

        llm_report: dict[str, Any]
        if run_llm:
            llm_report = self._run_local_llm(model=llm_model, endpoint=ollama_endpoint, timeout_sec=timeout_sec)
        else:
            llm_report = {"available": False, "skipped": True, "success_rate": 0.0}

        mcp_report: dict[str, Any]
        if run_mcp:
            try:
                mcp_report = asyncio.run(_run_mcp_suite(self.state_dir, timeout_sec=timeout_sec))
            except Exception as exc:
                mcp_report = {"available": False, "error": str(exc), "success_rate": 0.0}
        else:
            mcp_report = {"available": False, "skipped": True, "success_rate": 0.0}

        red_team_report: dict[str, Any]
        if run_red_team:
            try:
                campaign = ConsciousnessRedTeamCampaign(self.state_dir)
                rt = campaign.run(
                    persist=False,
                    base_seed=max(0, int(red_team_seed)),
                    max_scenarios=max(0, int(red_team_max_scenarios)),
                    quick=bool(red_team_quick),
                )
                red_team_report = {
                    **dict(rt.report),
                    "available": True,
                }
            except Exception as exc:
                red_team_report = {
                    "available": False,
                    "error": str(exc),
                    "pass_ratio": 0.0,
                    "mean_robustness": 0.0,
                    "scenario_count": 0,
                }
        else:
            red_team_report = {
                "available": False,
                "skipped": True,
                "pass_ratio": 0.0,
                "mean_robustness": 0.0,
                "scenario_count": 0,
            }

        core_composites = [_safe_float((row.get("scores") or {}).get("composite")) for row in core_runs]
        core_score = sum(core_composites) / max(len(core_composites), 1)
        trial_rci_delta = [
            _safe_float((row.get("delta") or {}).get("rci_delta"), default=0.0) for row in trials
        ]
        trial_score = _clamp(0.5 + (sum(trial_rci_delta) / max(len(trial_rci_delta), 1)), 0.0, 1.5)
        llm_score = _safe_float(llm_report.get("success_rate"), default=0.0)
        mcp_score = _safe_float(mcp_report.get("success_rate"), default=0.0)
        red_team_pass_ratio = _safe_float(red_team_report.get("pass_ratio"), default=0.0)
        red_team_robustness = _safe_float(red_team_report.get("mean_robustness"), default=0.0)
        red_team_score = _clamp((0.6 * red_team_pass_ratio) + (0.4 * red_team_robustness), 0.0, 1.0)

        integrated_score = _weighted_score(
            [
                (float(core_score), 0.45, True),
                (float(trial_score), 0.20, True),
                (float(llm_score), 0.20, bool(run_llm)),
                (float(mcp_score), 0.15, bool(run_mcp)),
                (float(red_team_score), 0.15, bool(run_red_team)),
            ]
        )

        latest = self.latest()
        baseline_score = None
        delta_score = None
        improved = None
        if isinstance(latest, Mapping):
            baseline_score = _safe_float((latest.get("scores") or {}).get("integrated"), default=0.0)
            delta_score = round(integrated_score - baseline_score, 6)
            improved = bool(delta_score >= -0.01)

        benchmark_id = f"integrated_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
        report = {
            "benchmark_id": benchmark_id,
            "timestamp": _now_iso(),
            "state_dir": str(self.state_dir),
            "config": {
                "rounds": rounds,
                "bench_ticks": bench_ticks,
                "trial_ticks": trial_ticks,
                "run_mcp": run_mcp,
                "run_llm": run_llm,
                "run_red_team": run_red_team,
                "red_team_quick": bool(red_team_quick),
                "red_team_max_scenarios": max(0, int(red_team_max_scenarios)),
                "red_team_seed": max(0, int(red_team_seed)),
                "llm_model": llm_model,
                "ollama_endpoint": ollama_endpoint,
            },
            "core_runs": core_runs,
            "trials": trials,
            "local_llm": llm_report,
            "mcp_runtime": mcp_report,
            "red_team": red_team_report,
            "scores": {
                "core_score": round(core_score, 6),
                "trial_score": round(float(trial_score), 6),
                "llm_score": round(float(llm_score), 6),
                "mcp_score": round(float(mcp_score), 6),
                "red_team_score": round(float(red_team_score), 6),
                "integrated": integrated_score,
                "baseline": baseline_score,
                "delta": delta_score,
            },
            "gates": {
                "core_score_min": core_score >= 0.35,
                "trial_score_min": trial_score >= 0.35,
                "llm_available": bool(llm_report.get("available")),
                "mcp_available": bool(mcp_report.get("available")),
                "red_team_available": bool(red_team_report.get("available")) if run_red_team else True,
                "llm_success_min": llm_score >= 0.5 if run_llm else True,
                "mcp_success_min": mcp_score >= 0.75 if run_mcp else True,
                "red_team_pass_min": red_team_pass_ratio >= 0.75 if run_red_team else True,
                "red_team_robustness_min": red_team_robustness >= 0.70 if run_red_team else True,
                "non_regression": improved if improved is not None else True,
            },
        }

        events.append(
            self.state_dir,
            "benchmark.integrated",
            {
                "benchmark_id": benchmark_id,
                "scores": report["scores"],
                "gates": report["gates"],
            },
            tags=["consciousness", "benchmark", "integrated"],
        )

        path: Optional[Path] = None
        if persist:
            path = _report_dir() / f"{benchmark_id}.json"
            path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            report["report_path"] = str(path)
        return IntegratedBenchmarkResult(benchmark_id=benchmark_id, report_path=path, report=report)

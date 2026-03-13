from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agent_forge.autonomy.supervisor import AutonomySupervisor
from agent_forge.core import events as E
from agent_forge.core import scheduler as SCH
from agent_forge.core import state as S


@dataclass
class _FakeSearchResult:
    source: str
    id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)


class _FakeBridge:
    def unified_search(self, query: str, limit: int = 10):
        return [
            _FakeSearchResult(
                source="knowledge",
                id="k1",
                content="consciousness benchmark latency memory knowledge bridge",
                score=0.9,
                metadata={"tags": ["benchmark", "latency", "memory"]},
            )
        ][:limit]


class _FakeMemory:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def remember(self, content: str, **kwargs):
        self.rows.append({"content": content, "kwargs": kwargs})
        return "mem-1"


class _FakeGraphRAG:
    def native_report_summary(self, limit: int = 5):
        return {
            "count": 2,
            "average_quality_score": 0.42,
            "average_rating": 3.0,
            "weak_communities": 1,
            "top_community": "code_forge",
            "reports": [
                {
                    "community": "code_forge",
                    "title": "Code Forge Community",
                    "summary": "2 nodes with benchmark failure evidence",
                    "rating": 3,
                    "quality_score": 0.42,
                    "quality_band": "weak",
                }
            ][:limit],
        }

    def native_artifact_summary(self, limit: int = 10):
        return {
            "count": 1,
            "benchmark_failures": 1,
            "drift_warning_artifacts": 1,
            "artifacts": [
                {
                    "artifact_path": "reports/code_forge/benchmark.json",
                    "kind": "code_forge_benchmark",
                    "benchmark_gate_pass": False,
                    "drift_warning_count": 2,
                }
            ][:limit],
        }


def test_supervisor_seeds_autonomous_mission(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    S.migrate(state_dir)

    fake_memory = _FakeMemory()
    supervisor = AutonomySupervisor(
        state_dir,
        repo_root=repo_root,
        bridge=_FakeBridge(),
        memory_system=fake_memory,
        graphrag=_FakeGraphRAG(),
        config={
            "enabled": True,
            "context_query": "consciousness latency benchmark",
            "policy": {"max_active_goals": 1, "allowed_templates": ["consciousness_guard"]},
            "missions": [
                {
                    "id": "consciousness_guard",
                    "title": "Autonomy: consciousness guard loop",
                    "drive": "integrity",
                    "template": "consciousness_guard",
                    "priority": 1.0,
                    "query": "consciousness benchmark latency memory",
                    "vars": {"benchmark_ticks": "3"},
                }
            ],
        },
    )

    payload = supervisor.tick(beat_count=7)

    goals = S.list_goals(state_dir)
    plans = S.list_plans(state_dir)
    steps = S.list_steps(state_dir)
    events = E.iter_events(state_dir, limit=None)

    assert payload["status"] == "selected"
    assert payload["mission_id"] == "consciousness_guard"
    assert len(goals) == 1
    assert len(plans) == 1
    assert len(steps) == 2
    assert plans[0].meta["template"] == "consciousness_guard"
    assert plans[0].meta["cwd"] == str(repo_root.resolve())
    assert any(evt.get("type") == "autonomy.mission_selected" for evt in events)
    selected = next(evt for evt in events if evt.get("type") == "autonomy.mission_selected")
    assert selected["data"]["report_count"] == 2
    assert selected["data"]["artifact_count"] == 1
    assert fake_memory.rows


def test_supervisor_blocks_disallowed_template(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    S.migrate(state_dir)

    supervisor = AutonomySupervisor(
        state_dir,
        repo_root=repo_root,
        bridge=_FakeBridge(),
        memory_system=_FakeMemory(),
        graphrag=_FakeGraphRAG(),
        config={
            "enabled": True,
            "policy": {"max_active_goals": 1, "allowed_templates": ["lint"]},
            "missions": [
                {
                    "id": "blocked_guard",
                    "title": "Autonomy: blocked guard loop",
                    "drive": "integrity",
                    "template": "consciousness_guard",
                    "priority": 1.0,
                }
            ],
        },
    )

    payload = supervisor.tick(beat_count=1)
    events = E.iter_events(state_dir, limit=None)

    assert payload["status"] == "idle"
    assert S.list_goals(state_dir) == []
    assert any(evt.get("type") == "autonomy.mission_blocked" for evt in events)


def test_scheduler_act_uses_plan_cwd(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    work_dir = tmp_path / "repo"
    work_dir.mkdir()
    S.migrate(state_dir)
    SCH.STATE_DIR = str(state_dir)

    goal = S.add_goal(state_dir, "Autonomy: lint", "integrity")
    SCH.create_plan_for_goal(
        str(state_dir),
        goal,
        template="lint",
        meta={"cwd": str(work_dir.resolve())},
    )
    step = S.list_steps(state_dir)[0]
    captured: dict[str, str] = {}

    def _fake_run_step(base_dir, step_id, cmd, *, cwd, budget_s, template=None):
        captured["cwd"] = cwd
        captured["template"] = template
        return {"status": "ok", "rc": 0}

    monkeypatch.setattr("agent_forge.core.scheduler.run_step", _fake_run_step)

    result = SCH.act({}, step)

    assert result["status"] == "ok"
    assert captured["cwd"] == str(work_dir.resolve())
    assert captured["template"] == "lint"


def test_supervisor_hygiene_scoring_uses_report_and_artifact_risk(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    S.migrate(state_dir)

    supervisor = AutonomySupervisor(
        state_dir,
        repo_root=repo_root,
        bridge=_FakeBridge(),
        memory_system=_FakeMemory(),
        graphrag=_FakeGraphRAG(),
        config={
            "enabled": True,
            "context_query": "repo benchmark drift artifact report",
            "policy": {"max_active_goals": 1, "allowed_templates": ["hygiene", "consciousness_guard"]},
            "missions": [
                {
                    "id": "consciousness_guard",
                    "title": "Autonomy: consciousness guard loop",
                    "drive": "integrity",
                    "template": "consciousness_guard",
                    "priority": 1.0,
                    "query": "consciousness memory knowledge",
                },
                {
                    "id": "repo_hygiene",
                    "title": "Autonomy: repo hygiene sweep",
                    "drive": "integrity",
                    "template": "hygiene",
                    "priority": 0.65,
                    "query": "repo benchmark drift artifact report",
                },
            ],
        },
    )

    payload = supervisor.tick(beat_count=2)

    assert payload["mission_id"] == "repo_hygiene"
    assert payload["benchmark_failures"] == 1
    assert payload["average_report_quality"] == 0.42


def test_supervisor_reads_local_agent_runtime_state(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "runtime" / "local_mcp_agent").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "runtime" / "local_mcp_agent" / "status.json").write_text(
        '{"status":"timeout","profile":"observer","tool_calls":1,"resource_count":1,"mcp_transport":"stdio"}',
        encoding="utf-8",
    )
    (repo_root / "data" / "runtime" / "directory_docs_status.json").write_text(
        '{"missing_readme_count":4,"coverage_ratio":0.97,"missing_examples":["code_forge/src/code_forge/library"]}',
        encoding="utf-8",
    )
    S.migrate(state_dir)

    supervisor = AutonomySupervisor(
        state_dir,
        repo_root=repo_root,
        bridge=_FakeBridge(),
        memory_system=_FakeMemory(),
        graphrag=_FakeGraphRAG(),
        config={
            "enabled": True,
            "policy": {"max_active_goals": 1, "allowed_templates": ["hygiene"]},
            "missions": [
                {
                    "id": "repo_hygiene",
                    "title": "Autonomy: repo hygiene sweep",
                    "drive": "integrity",
                    "template": "hygiene",
                    "priority": 0.5,
                    "query": "repo benchmark drift artifact report",
                }
            ],
        },
    )

    payload = supervisor.tick(beat_count=3)
    assert payload["status"] == "selected"
    assert payload["local_agent_status"] == "timeout"
    assert payload["local_agent_tool_calls"] == 1
    assert payload["directory_docs_missing"] == 4
    assert payload["directory_docs_coverage"] == 0.97

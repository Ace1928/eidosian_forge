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
            "reports": [
                {
                    "community": "code_forge",
                    "title": "Code Forge Community",
                    "summary": "benchmark drift triage coverage",
                }
            ][:limit],
        }

    def native_artifact_summary(self, limit: int = 10):
        return {
            "count": 1,
            "items": [
                {
                    "kind": "code_forge_provenance_registry",
                    "artifact_path": "data/code_forge/cycle/run_001/provenance_registry.json",
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
        config={
            "enabled": True,
            "policy": {"max_active_goals": 1, "allowed_templates": ["hygiene"]},
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


def test_supervisor_context_includes_native_report_and_artifact_tokens(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    S.migrate(state_dir)

    supervisor = AutonomySupervisor(
        state_dir,
        repo_root=repo_root,
        bridge=_FakeBridge(),
        memory_system=_FakeMemory(),
        config={"context_query": "benchmark drift triage"},
    )
    monkeypatch.setattr(supervisor, "_load_graphrag", lambda: _FakeGraphRAG())

    packet = supervisor._context_packet()

    assert packet["report_summary"]["count"] == 2
    assert packet["artifact_summary"]["count"] == 1
    tokens = set(packet["tokens"])
    assert "benchmark" in tokens
    assert "triage" in tokens
    assert "provenance_registry.json" in tokens

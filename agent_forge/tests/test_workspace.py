from agent_forge.core import events
from agent_forge.core import workspace


def test_workspace_summary_detects_ignition(tmp_path):
    base = tmp_path / "state"
    events.append(base, "workspace.broadcast", {"source": "a", "payload": {"x": 1}})
    events.append(base, "workspace.broadcast", {"source": "b", "payload": {"x": 2}})
    summary = workspace.summary(base, window_seconds=60, min_sources=2)
    assert summary["event_count"] == 2
    assert summary["ignition_count"] >= 1

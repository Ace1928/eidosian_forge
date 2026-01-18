from planners.registry import choose
from planners.htn import materialize
from core.contracts import Goal


def test_choose_and_materialize_lint():
    g = Goal("g2", "Lint: codebase", "integrity", "2025-01-01T00:00:00Z")
    kind, meta = choose(g)
    assert kind == "htn" and meta["template"] == "lint"
    steps = materialize(meta["template"], g.title)
    names = [s["name"] for s in steps]
    assert names[:2] == ["lint-fix", "lint-check"]
    assert steps[0]["cmd"][0] == "ruff"

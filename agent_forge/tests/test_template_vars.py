from planners.registry import choose
from planners.htn import materialize
from core.contracts import Goal


def test_template_vars_substitution():
    g = Goal("g3", "Hygiene: smoke", "integrity", "2025-01-01T00:00:00Z")
    kind, meta = choose(g)
    assert meta.get("vars")
    steps = materialize(meta["template"], g.title, vars={"pytest_args": "-q -k smoke"})
    test_step = [s for s in steps if s["name"] == "test"][0]
    assert test_step["cmd"][:3] == ["pytest", "-q", "-k"]

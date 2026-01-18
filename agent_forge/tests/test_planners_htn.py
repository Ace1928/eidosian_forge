from planners.htn import materialize
from planners.registry import choose
from core.contracts import Goal
def test_materialize_hygiene(tmp_path):
    g = Goal("g1","Hygiene: format & smoke","integrity","2025-01-01T00:00:00Z")
    kind, meta = choose(g)
    assert kind == "htn" and meta["template"]=="hygiene"
    steps = materialize(meta["template"], g.title)
    assert [s["name"] for s in steps] == ["format","test"]
    assert steps[0]["cmd"][0] == "ruff"

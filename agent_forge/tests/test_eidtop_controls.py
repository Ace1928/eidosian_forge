import importlib.util
from pathlib import Path


def _load():
    loader = importlib.machinery.SourceFileLoader("eidtop", str(Path("bin/eidtop")))
    spec = importlib.util.spec_from_loader("eidtop", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_render_pause_and_stats():
    mod = _load()
    model = {
        "events": [
            {
                "ts": "2024-01-01T00:00:00Z",
                "data": {"cpu_pct": 1.0, "rss_bytes": 1000},
            }
        ],
        "metrics": {"process.cpu_pct": [("t1", 1.0), ("t2", 2.0)]},
        "paused": True,
        "focus": "process.cpu_pct",
    }
    lines = mod.render(model, width=200)
    assert any("PAUSED" in line for line in lines)
    assert any("min" in line and "avg" in line and "max" in line for line in lines)


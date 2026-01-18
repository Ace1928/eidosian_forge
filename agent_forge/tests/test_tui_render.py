import runpy

mod = runpy.run_path("bin/eidtop")
render = mod["render"]


def test_render_basic():
    model = {
        "events": [{"ts": "2020", "data": {"cpu_pct": 12.5, "rss_bytes": 1024, "load1": 0.5, "tick_secs": 1.0}}],
        "metrics": {},
    }
    lines = render(model)
    assert any("cpu" in line and "12.5" in line for line in lines)
    assert any("rss" in line for line in lines)
    assert any("load1" in line for line in lines)
    assert any("tick" in line for line in lines)
    assert lines[-1].startswith("[q]")

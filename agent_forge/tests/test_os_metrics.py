import itertools

from core import os_metrics as OM


def test_process_stats_keys():
    stats = OM.process_stats()
    assert {"rss_bytes", "cpu_user_s", "cpu_sys_s"}.issubset(stats)
    for v in stats.values():
        assert v is None or v >= 0


def test_system_stats_keys():
    stats = OM.system_stats()
    assert "load1" in stats and "mem_total_kb" in stats
    for v in stats.values():
        assert v is None or isinstance(v, (int, float))


def test_cpu_percent(monkeypatch):
    vals = itertools.cycle([(1.0, 0.0), (1.5, 0.0)])
    monkeypatch.setattr(OM, '_read_proc_stat', lambda: next(vals))
    times = iter([1.0, 2.0])
    monkeypatch.setattr(OM.time, 'monotonic', lambda: next(times))
    cp = OM.CpuPercent()
    assert cp.sample() is None
    pct = cp.sample()
    assert pct is not None and 0.0 <= pct <= 100.0

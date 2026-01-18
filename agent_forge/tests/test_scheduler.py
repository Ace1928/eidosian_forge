import random

from core.scheduler import BeatCfg, StopToken, run_loop


def test_run_loop_max_beats(monkeypatch):
    calls = []

    def beat():
        calls.append(1)

    monkeypatch.setattr('time.sleep', lambda s: None)
    cfg = BeatCfg(tick_secs=0.0, max_beats=3)
    run_loop(cfg, beat, stop=StopToken())
    assert len(calls) == 3


def test_run_loop_backoff(monkeypatch):
    seq = []
    monkeypatch.setattr('time.sleep', lambda s: seq.append(s))
    calls = {'n': 0}

    def beat():
        calls['n'] += 1
        if calls['n'] in (2, 3, 4):
            raise RuntimeError

    cfg = BeatCfg(tick_secs=0.0, max_beats=2, max_backoff_secs=4)
    run_loop(cfg, beat, stop=StopToken())
    assert seq[:3] == [1.0, 2.0, 4.0]


def test_run_loop_jitter(monkeypatch):
    sleeps = []
    monkeypatch.setattr('time.sleep', lambda s: sleeps.append(s))
    monkeypatch.setattr(random, 'uniform', lambda a, b: 0.0)

    cfg = BeatCfg(tick_secs=3.0, jitter_ms=100, max_beats=1)
    run_loop(cfg, lambda: None, stop=StopToken())
    assert sleeps == [3.0]

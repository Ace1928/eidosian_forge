from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agent_forge.consciousness.metrics import (
    directionality_asymmetry,
    effective_connectivity,
    response_complexity,
    self_stability,
)


def _ts(base: datetime, sec: float) -> str:
    return (base + timedelta(seconds=sec)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_response_complexity_v2_keys_present() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "type": "sense.percept",
            "ts": _ts(base, 0),
            "corr_id": "c1",
            "parent_id": "p1",
            "data": {"novelty": 0.6, "source_module": "sense"},
        },
        {
            "type": "attn.candidate",
            "ts": _ts(base, 0.2),
            "corr_id": "c1",
            "parent_id": "p1",
            "data": {"candidate_id": "cand-1", "source_module": "attention"},
        },
        {
            "type": "policy.action",
            "ts": _ts(base, 0.4),
            "corr_id": "c1",
            "parent_id": "p1",
            "data": {"selected_candidate_id": "cand-1", "source_module": "policy"},
        },
    ]

    out = response_complexity(rows)
    assert "rci" in out
    assert "rci_v2" in out
    assert "event_type_entropy" in out
    assert "source_entropy" in out
    assert "integration_proxy" in out
    assert out["rci_v2"] >= 0.0


def test_effective_connectivity_uses_winner_linked_reactions() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    winner = {
        "type": "workspace.broadcast",
        "ts": _ts(base, 0.0),
        "corr_id": "c2",
        "parent_id": "p2",
        "data": {
            "source": "workspace_competition",
            "payload": {
                "kind": "GW_WINNER",
                "content": {
                    "candidate_id": "cand-2",
                    "winner_candidate_id": "cand-2",
                    "source_module": "sense",
                },
                "links": {
                    "corr_id": "c2",
                    "parent_id": "p2",
                    "candidate_id": "cand-2",
                    "winner_candidate_id": "cand-2",
                    "memory_ids": [],
                },
            },
        },
    }
    policy = {
        "type": "policy.action",
        "ts": _ts(base, 0.5),
        "corr_id": "c2",
        "parent_id": "p2",
        "data": {"selected_candidate_id": "cand-2", "source_module": "policy"},
    }
    report = {
        "type": "workspace.broadcast",
        "ts": _ts(base, 1.0),
        "corr_id": "c2",
        "parent_id": "p2",
        "data": {
            "source": "report",
            "payload": {
                "kind": "REPORT",
                "content": {"summary": {"winner_candidate_id": "cand-2"}},
                "links": {
                    "corr_id": "c2",
                    "parent_id": "p2",
                    "winner_candidate_id": "cand-2",
                    "candidate_id": "",
                    "memory_ids": [],
                },
            },
        },
    }

    out = effective_connectivity([winner, policy, report], reaction_window_secs=2.0)
    assert out["node_count"] >= 3
    assert out["edge_count"] >= 2
    assert out["density"] > 0.0
    assert out["workspace_centrality"] >= 0.0


def test_directionality_asymmetry_exposes_pair_metrics() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(6):
        t = float(i)
        rows.append(
            {
                "type": "policy.action",
                "ts": _ts(base, t),
                "corr_id": f"c{i}",
                "parent_id": f"p{i}",
                "data": {"source_module": "policy"},
            }
        )
        rows.append(
            {
                "type": "report.self_report",
                "ts": _ts(base, t + 0.2),
                "corr_id": f"c{i}",
                "parent_id": f"p{i}",
                "data": {"source_module": "report"},
            }
        )

    out = directionality_asymmetry(rows)
    assert out["window_count"] >= 1
    assert out["pair_count"] >= 1
    assert out["mean_abs_asymmetry"] >= 0.0
    assert out["pairs"]


def test_self_stability_score_penalizes_high_variance() -> None:
    steady = [
        {
            "type": "metrics.sample",
            "ts": "2026-01-01T00:00:00Z",
            "data": {"key": "consciousness.agency", "value": 0.8},
        },
        {
            "type": "metrics.sample",
            "ts": "2026-01-01T00:00:01Z",
            "data": {"key": "consciousness.agency", "value": 0.81},
        },
        {
            "type": "metrics.sample",
            "ts": "2026-01-01T00:00:02Z",
            "data": {"key": "consciousness.boundary_stability", "value": 0.82},
        },
        {
            "type": "metrics.sample",
            "ts": "2026-01-01T00:00:03Z",
            "data": {"key": "consciousness.boundary_stability", "value": 0.83},
        },
    ]
    volatile = [
        {
            "type": "metrics.sample",
            "ts": "2026-01-01T00:00:00Z",
            "data": {"key": "consciousness.agency", "value": 0.1},
        },
        {
            "type": "metrics.sample",
            "ts": "2026-01-01T00:00:01Z",
            "data": {"key": "consciousness.agency", "value": 0.9},
        },
        {
            "type": "metrics.sample",
            "ts": "2026-01-01T00:00:02Z",
            "data": {"key": "consciousness.boundary_stability", "value": 0.05},
        },
        {
            "type": "metrics.sample",
            "ts": "2026-01-01T00:00:03Z",
            "data": {"key": "consciousness.boundary_stability", "value": 0.95},
        },
    ]

    stable_out = self_stability(steady)
    volatile_out = self_stability(volatile)

    assert stable_out["stability_score"] > volatile_out["stability_score"]

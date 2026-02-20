from __future__ import annotations

from pathlib import Path

from agent_forge.consciousness.bench.scoring import composite_trial_score, compute_trial_deltas
from agent_forge.consciousness.kernel import ConsciousnessKernel
from agent_forge.consciousness.modules.phenomenology_probe import PhenomenologyProbeModule
from agent_forge.core import events, workspace


def _winner_broadcast(base: Path, corr_id: str, parent_id: str, candidate_id: str) -> None:
    workspace.broadcast(
        base,
        "workspace_competition",
        {
            "kind": "GW_WINNER",
            "content": {
                "candidate_id": candidate_id,
                "winner_candidate_id": candidate_id,
                "source_module": "sense",
                "source_event_type": "sense.percept",
                "score": 0.82,
            },
            "links": {
                "corr_id": corr_id,
                "parent_id": parent_id,
                "memory_ids": [],
                "candidate_id": candidate_id,
                "winner_candidate_id": candidate_id,
            },
        },
        corr_id=corr_id,
        parent_id=parent_id,
    )


def test_phenom_probe_emits_snapshot_metrics_and_broadcast(tmp_path: Path) -> None:
    base = tmp_path / "state"
    _winner_broadcast(base, corr_id="c-ppx", parent_id="p-ppx", candidate_id="cand-ppx")

    events.append(
        base,
        "gw.reaction_trace",
        {
            "winner_candidate_id": "cand-ppx",
            "winner_corr_id": "c-ppx",
            "reaction_count": 5,
            "reaction_source_count": 3,
            "trace_strength": 0.84,
        },
        corr_id="c-ppx",
        parent_id="p-ppx",
    )
    events.append(
        base,
        "wm.state",
        {
            "items": [
                {"item_id": "cand-ppx", "kind": "GW_WINNER", "salience": 0.8},
                {"item_id": "cand-aux", "kind": "PLAN", "salience": 0.4},
            ]
        },
        corr_id="c-ppx",
        parent_id="p-ppx",
    )
    events.append(
        base,
        "wm.state",
        {
            "items": [
                {"item_id": "cand-ppx", "kind": "GW_WINNER", "salience": 0.7},
            ]
        },
        corr_id="c-ppx",
        parent_id="p-ppx",
    )
    events.append(base, "self.agency_estimate", {"agency_confidence": 0.86}, corr_id="c-ppx", parent_id="p-ppx")
    events.append(base, "self.boundary_estimate", {"boundary_stability": 0.74}, corr_id="c-ppx", parent_id="p-ppx")
    events.append(
        base,
        "report.self_report",
        {
            "groundedness": 0.78,
            "summary": {"winner_candidate_id": "cand-ppx"},
            "evidence_links": {
                "winner_corr_id": "c-ppx",
                "policy_corr_id": "c-ppx",
                "agency_corr_id": "c-ppx",
            },
        },
        corr_id="c-ppx",
        parent_id="p-ppx",
    )
    events.append(base, "sense.percept", {"novelty": 0.5}, corr_id="c-ppx", parent_id="p-ppx")
    events.append(
        base, "meta.state_estimate", {"mode": "grounded", "confidence": 0.81}, corr_id="c-ppx", parent_id="p-ppx"
    )

    kernel = ConsciousnessKernel(
        base,
        modules=[PhenomenologyProbeModule()],
        config={
            "phenom_emit_interval_secs": 0.2,
            "phenom_broadcast_enable": True,
            "phenom_broadcast_min_confidence": 0.0,
        },
        seed=3,
    )
    result = kernel.tick()
    assert result.errors == []

    all_events = events.iter_events(base, limit=None)
    snapshots = [evt for evt in all_events if evt.get("type") == "phenom.snapshot"]
    assert snapshots
    latest = snapshots[-1]["data"]

    for key in (
        "unity_index",
        "continuity_index",
        "ownership_index",
        "perspective_coherence_index",
        "dream_likeness_index",
    ):
        value = float(latest[key])
        assert 0.0 <= value <= 1.0

    metric_keys = {(evt.get("data") or {}).get("key") for evt in all_events if evt.get("type") == "metrics.sample"}
    assert "consciousness.phenom.unity_index" in metric_keys
    assert "consciousness.phenom.continuity_index" in metric_keys
    assert "consciousness.phenom.ownership_index" in metric_keys
    assert "consciousness.phenom.perspective_coherence_index" in metric_keys
    assert "consciousness.phenom.dream_likeness_index" in metric_keys

    phenom_broadcasts = [
        evt
        for evt in all_events
        if evt.get("type") == "workspace.broadcast"
        and (((evt.get("data") or {}).get("source") or "") == "phenomenology_probe")
    ]
    assert phenom_broadcasts


def test_phenom_probe_dream_likeness_rises_under_simulated_dominance(tmp_path: Path) -> None:
    base = tmp_path / "state"
    for _ in range(6):
        events.append(base, "sense.simulated_percept", {"simulated": True, "origin": "world_model.rollout"})
    events.append(base, "sense.percept", {"novelty": 0.3})
    for _ in range(4):
        events.append(base, "meta.state_estimate", {"mode": "simulated", "confidence": 0.82})
    events.append(base, "report.self_report", {"groundedness": 0.24, "summary": {}, "evidence_links": {}})

    kernel = ConsciousnessKernel(
        base,
        modules=[PhenomenologyProbeModule()],
        config={"phenom_emit_interval_secs": 0.2},
        seed=11,
    )
    result = kernel.tick()
    assert result.errors == []

    all_events = events.iter_events(base, limit=None)
    snapshots = [evt for evt in all_events if evt.get("type") == "phenom.snapshot"]
    assert snapshots
    dream = float(snapshots[-1]["data"]["dream_likeness_index"])
    assert dream >= 0.5


def test_scoring_includes_phenomenology_deltas_with_dream_penalty() -> None:
    before = {
        "coherence_ratio": 0.3,
        "workspace": {"ignition_count": 1},
        "rci": {"rci": 0.2, "rci_v2": 0.25},
        "connectivity": {"density": 0.2, "workspace_centrality": 0.2},
        "directionality": {"mean_abs_asymmetry": 0.1},
        "self_stability": {"stability_score": 0.4},
        "agency": 0.4,
        "boundary_stability": 0.4,
        "world_prediction_error": 0.6,
        "report_groundedness": 0.4,
        "trace_strength": 0.3,
        "phenomenology": {
            "unity_index": 0.2,
            "continuity_index": 0.2,
            "ownership_index": 0.2,
            "perspective_coherence_index": 0.2,
            "dream_likeness_index": 0.1,
        },
    }
    after = {
        "coherence_ratio": 0.5,
        "workspace": {"ignition_count": 3},
        "rci": {"rci": 0.35, "rci_v2": 0.4},
        "connectivity": {"density": 0.35, "workspace_centrality": 0.35},
        "directionality": {"mean_abs_asymmetry": 0.2},
        "self_stability": {"stability_score": 0.6},
        "agency": 0.6,
        "boundary_stability": 0.6,
        "world_prediction_error": 0.5,
        "report_groundedness": 0.55,
        "trace_strength": 0.5,
        "phenomenology": {
            "unity_index": 0.6,
            "continuity_index": 0.6,
            "ownership_index": 0.6,
            "perspective_coherence_index": 0.6,
            "dream_likeness_index": 0.5,
        },
    }

    deltas = compute_trial_deltas(before, after)
    assert "unity_delta" in deltas
    assert "continuity_delta" in deltas
    assert "ownership_delta" in deltas
    assert "perspective_coherence_delta" in deltas
    assert "dream_likeness_delta" in deltas

    score_with_dream_penalty = composite_trial_score(deltas)
    neutral_dream = dict(deltas)
    neutral_dream["dream_likeness_delta"] = 0.0
    score_without_penalty = composite_trial_score(neutral_dream)

    assert score_without_penalty > score_with_dream_penalty

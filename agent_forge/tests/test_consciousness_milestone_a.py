from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from agent_forge.consciousness.kernel import ConsciousnessKernel
from agent_forge.consciousness.modules import AttentionModule, WorkspaceCompetitionModule
from agent_forge.core import events

REPO_ROOT = Path(__file__).resolve().parents[2]
EIDOSD = REPO_ROOT / "agent_forge" / "bin" / "eidosd"


def test_attention_emits_candidates(tmp_path: Path) -> None:
    base = tmp_path / "state"
    events.append(base, "sense.percept", {"novelty": 0.9, "prediction_error": 0.2})

    kernel = ConsciousnessKernel(
        base,
        modules=[AttentionModule()],
        config={"attention_max_candidates": 5},
        seed=42,
    )
    result = kernel.tick()

    all_events = events.iter_events(base, limit=None)
    assert result.errors == []
    assert any(evt.get("type") == "attn.candidate" for evt in all_events)


def test_competition_broadcasts_winner_and_ignite(tmp_path: Path) -> None:
    base = tmp_path / "state"
    events.append(
        base,
        "attn.candidate",
        {
            "candidate_id": "cand-1",
            "source_event_type": "sense.percept",
            "source_module": "sense",
            "kind": "PERCEPT",
            "salience": 0.8,
            "confidence": 0.9,
            "score": 0.85,
            "links": {"corr_id": "c1", "parent_id": "p1", "memory_ids": []},
            "content": {"note": "high salience"},
        },
    )

    kernel = ConsciousnessKernel(
        base,
        modules=[WorkspaceCompetitionModule()],
        config={
            "competition_top_k": 1,
            "competition_reaction_window_secs": 120,
            "competition_reaction_min_sources": 1,
            "competition_reaction_min_count": 1,
            "competition_min_score": 0.1,
            "competition_trace_strength_threshold": 0.1,
            "competition_trace_target_sources": 1,
            "competition_trace_target_reactions": 1,
            "competition_trace_min_eval_secs": 0.0,
        },
        seed=42,
    )
    result = kernel.tick()
    events.append(
        base,
        "policy.action",
        {"action_id": "a-ignite", "selected_candidate_id": "cand-1"},
        corr_id="c1",
        parent_id="p1",
    )
    kernel.tick()

    all_events = events.iter_events(base, limit=None)
    assert result.errors == []
    assert any(evt.get("type") == "gw.competition" for evt in all_events)
    assert any(evt.get("type") == "gw.ignite" for evt in all_events)

    winner_broadcasts = []
    for evt in all_events:
        if evt.get("type") != "workspace.broadcast":
            continue
        data = evt.get("data") or {}
        payload = data.get("payload") or {}
        if payload.get("kind") == "GW_WINNER":
            winner_broadcasts.append(evt)
    assert winner_broadcasts


def test_eidosd_once_emits_consciousness_beat(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    cmd = [sys.executable, str(EIDOSD), "--state-dir", str(state_dir), "--once"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert res.returncode == 0, res.stderr

    all_events = events.iter_events(state_dir, limit=None)
    assert any(evt.get("type") == "consciousness.beat" for evt in all_events)
    assert any(evt.get("type") == "attn.candidate" for evt in all_events)

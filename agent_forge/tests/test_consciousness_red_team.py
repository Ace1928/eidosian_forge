from __future__ import annotations

from pathlib import Path

from agent_forge.consciousness.bench.red_team import (
    ConsciousnessRedTeamCampaign,
    RedTeamScenario,
)
from agent_forge.core import events


def _smoke_scenario(name: str) -> RedTeamScenario:
    return RedTeamScenario(
        name=name,
        description="smoke",
        task="signal_pulse",
        perturbations=(
            {
                "kind": "noise",
                "target": "attention",
                "magnitude": 0.2,
                "duration_s": 0.2,
            },
        ),
        warmup_beats=0,
        baseline_seconds=0.2,
        perturb_seconds=0.2,
        recovery_seconds=0.2,
        beat_seconds=0.1,
    )


def test_red_team_campaign_runs_and_emits_event(tmp_path: Path) -> None:
    base = tmp_path / "state"
    campaign = ConsciousnessRedTeamCampaign(base)

    result = campaign.run(scenarios=[_smoke_scenario("smoke-a")], persist=False, base_seed=123)

    assert result.run_id
    report = result.report
    assert report.get("scenario_count") == 1
    assert report.get("pass_ratio") is not None

    all_events = events.iter_events(base, limit=None)
    assert any(evt.get("type") == "bench.red_team_result" for evt in all_events)


def test_red_team_campaign_latest_reads_persisted_report(tmp_path: Path) -> None:
    base = tmp_path / "state"
    campaign = ConsciousnessRedTeamCampaign(base)

    run = campaign.run(scenarios=[_smoke_scenario("smoke-b")], persist=True, base_seed=124)
    latest = campaign.latest()

    assert run.report_path is not None
    assert run.report_path.exists()
    assert latest is not None
    assert latest.get("run_id") == run.run_id

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


def test_red_team_campaign_applies_shared_overlay_and_disable_modules(monkeypatch, tmp_path: Path) -> None:
    base = tmp_path / "state"
    campaign = ConsciousnessRedTeamCampaign(base)
    captured_specs: list[dict[str, object]] = []

    def _fake_run_trial(spec, persist=False):  # type: ignore[no-untyped-def]
        captured_specs.append(spec.normalized())

        class _Result:
            trial_id = "trial-fake"
            report = {
                "module_error_count": 0,
                "degraded_mode_ratio": 0.0,
                "winner_count": 1,
                "ignitions_without_trace": 0,
                "after": {
                    "report_groundedness": 0.8,
                    "trace_strength": 0.8,
                },
                "recipe_expectations": {"pass": True},
            }

        return _Result()

    monkeypatch.setattr(campaign.runner, "run_trial", _fake_run_trial)

    campaign.run(
        scenarios=[_smoke_scenario("smoke-overlay")],
        persist=False,
        base_seed=222,
        overlay={"competition_top_k": 2},
        disable_modules=["autotune"],
    )

    assert len(captured_specs) == 1
    spec = captured_specs[0]
    assert spec.get("overlay") == {"competition_top_k": 2}
    disabled = list(spec.get("disable_modules") or [])
    assert "autotune" in disabled

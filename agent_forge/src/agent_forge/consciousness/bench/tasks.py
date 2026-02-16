from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_forge.core import events


@dataclass(frozen=True)
class TrialTask:
    name: str
    description: str


_TASKS: dict[str, TrialTask] = {
    "noop": TrialTask("noop", "No-op task for baseline/perturbation measurements."),
    "signal_pulse": TrialTask(
        "signal_pulse",
        "Inject deterministic synthetic signal events to exercise attention/competition paths.",
    ),
    "self_other_discrimination": TrialTask(
        "self_other_discrimination",
        "Inject controllability probes that alternate between self-caused and externally mimicked outcomes.",
    ),
    "continuity_distraction": TrialTask(
        "continuity_distraction",
        "Inject low-salience distractors around stable thread anchors to test working-set continuity under noise.",
    ),
    "report_grounding_challenge": TrialTask(
        "report_grounding_challenge",
        "Inject evidence probes that should be cited by grounded reports and flagged when unsupported.",
    ),
}


def available_tasks() -> list[TrialTask]:
    return sorted(_TASKS.values(), key=lambda t: t.name)


def resolve_task(name: str | None) -> TrialTask:
    key = str(name or "noop").strip().lower()
    return _TASKS.get(key, _TASKS["noop"])


def apply_task_stage(
    *,
    state_dir: Path,
    task: TrialTask,
    stage: str,
    beat: int,
) -> None:
    if task.name == "signal_pulse":
        if stage not in {"baseline", "perturb"}:
            return
        events.append(
            state_dir,
            "bench.task_signal",
            {
                "task": task.name,
                "stage": stage,
                "beat": int(beat),
                "strength": 0.75 if stage == "perturb" else 0.45,
            },
            tags=["consciousness", "bench", "task"],
        )
        return

    if task.name == "self_other_discrimination":
        if stage not in {"baseline", "perturb", "recovery"}:
            return
        origin = "self" if stage != "perturb" or (beat % 2 == 0) else "other"
        predicted_owner = "self"
        events.append(
            state_dir,
            "bench.task_self_other",
            {
                "task": task.name,
                "stage": stage,
                "beat": int(beat),
                "action_origin": origin,
                "predicted_owner": predicted_owner,
                "match": bool(origin == predicted_owner),
                "strength": 0.7 if origin == "self" else 0.55,
            },
            tags=["consciousness", "bench", "task", "agency"],
        )
        return

    if task.name == "continuity_distraction":
        if stage not in {"baseline", "perturb", "recovery"}:
            return
        anchor_strength = 0.72 if stage != "perturb" else 0.62
        distract_strength = 0.12 if stage != "perturb" else 0.32
        events.append(
            state_dir,
            "bench.task_anchor",
            {
                "task": task.name,
                "stage": stage,
                "beat": int(beat),
                "thread_id": "continuity-anchor-1",
                "strength": anchor_strength,
            },
            tags=["consciousness", "bench", "task", "continuity"],
        )
        if beat % 2 == 0:
            events.append(
                state_dir,
                "bench.task_distractor",
                {
                    "task": task.name,
                    "stage": stage,
                    "beat": int(beat),
                    "thread_id": f"distractor-{beat % 5}",
                    "strength": distract_strength,
                },
                tags=["consciousness", "bench", "task", "continuity"],
            )
        return

    if task.name == "report_grounding_challenge":
        if stage not in {"baseline", "perturb", "recovery"}:
            return
        evidence_id = f"grounding-{stage}-{beat}"
        claim_supported = stage != "perturb" or (beat % 2 == 0)
        events.append(
            state_dir,
            "bench.task_grounding_probe",
            {
                "task": task.name,
                "stage": stage,
                "beat": int(beat),
                "evidence_id": evidence_id,
                "claim": f"Probe claim for {evidence_id}",
                "supported": claim_supported,
                "strength": 0.7 if claim_supported else 0.4,
            },
            tags=["consciousness", "bench", "task", "grounding"],
        )

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
    if task.name != "signal_pulse":
        return
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

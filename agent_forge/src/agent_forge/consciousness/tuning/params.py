from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ParamKind = Literal["float", "int", "bool", "choice"]
SafetyClass = Literal["safe", "risky", "dangerous"]


@dataclass(frozen=True)
class ParamSpec:
    key: str
    kind: ParamKind
    default: Any
    min_value: float | int | None = None
    max_value: float | int | None = None
    choices: list[Any] | None = None
    safety: SafetyClass = "safe"
    description: str = ""


def default_param_specs() -> dict[str, ParamSpec]:
    # High-leverage, low-risk parameters only for initial self-tuning loops.
    specs = [
        ParamSpec(
            key="competition_top_k",
            kind="int",
            default=2,
            min_value=1,
            max_value=4,
            description="Number of winner candidates admitted per competition cycle.",
        ),
        ParamSpec(
            key="competition_min_score",
            kind="float",
            default=0.15,
            min_value=0.05,
            max_value=0.8,
            description="Minimum candidate score required for workspace broadcast.",
        ),
        ParamSpec(
            key="competition_trace_strength_threshold",
            kind="float",
            default=0.45,
            min_value=0.2,
            max_value=0.95,
            description="Ignition gate on winner-linked trace strength.",
        ),
        ParamSpec(
            key="competition_reaction_window_secs",
            kind="float",
            default=1.5,
            min_value=0.3,
            max_value=6.0,
            description="Reaction window for winner-linked ignition tracing.",
        ),
        ParamSpec(
            key="competition_cooldown_secs",
            kind="float",
            default=2.5,
            min_value=0.0,
            max_value=8.0,
            description="Cooldown duration for repeated winner signatures.",
        ),
        ParamSpec(
            key="world_belief_alpha",
            kind="float",
            default=0.22,
            min_value=0.02,
            max_value=0.8,
            description="EMA update rate for world-model belief features.",
        ),
        ParamSpec(
            key="world_error_broadcast_threshold",
            kind="float",
            default=0.55,
            min_value=0.1,
            max_value=0.95,
            description="Error threshold for world prediction broadcast.",
        ),
        ParamSpec(
            key="world_error_derivative_threshold",
            kind="float",
            default=0.2,
            min_value=0.01,
            max_value=0.9,
            description="Prediction-error derivative gate for surprise broadcasts.",
        ),
        ParamSpec(
            key="world_prediction_window",
            kind="int",
            default=120,
            min_value=24,
            max_value=600,
            description="World model temporal context window in events.",
        ),
        ParamSpec(
            key="meta_observation_window",
            kind="int",
            default=160,
            min_value=24,
            max_value=800,
            description="Observation depth for mode-classification dynamics.",
        ),
        ParamSpec(
            key="meta_emit_delta_threshold",
            kind="float",
            default=0.05,
            min_value=0.01,
            max_value=0.5,
            description="Minimum state-change delta required for meta emission.",
        ),
        ParamSpec(
            key="module_tick_periods.world_model",
            kind="int",
            default=1,
            min_value=1,
            max_value=8,
            description="World model beat period.",
        ),
        ParamSpec(
            key="module_tick_periods.meta",
            kind="int",
            default=2,
            min_value=1,
            max_value=10,
            description="Meta module beat period.",
        ),
        ParamSpec(
            key="module_tick_periods.report",
            kind="int",
            default=2,
            min_value=1,
            max_value=10,
            description="Report module beat period.",
        ),
        ParamSpec(
            key="module_tick_periods.phenomenology_probe",
            kind="int",
            default=3,
            min_value=1,
            max_value=12,
            description="Phenomenology probe beat period.",
        ),
    ]
    return {spec.key: spec for spec in specs}


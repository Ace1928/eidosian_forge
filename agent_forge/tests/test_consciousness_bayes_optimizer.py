from __future__ import annotations

import random

from agent_forge.consciousness.tuning.bayes_optimizer import (
    BayesParetoOptimizer,
    pareto_front,
)
from agent_forge.consciousness.tuning.params import default_param_specs


def test_pareto_front_filters_dominated_points() -> None:
    points = [
        {"coherence": 0.2, "groundedness": 0.1},
        {"coherence": 0.4, "groundedness": 0.3},
        {"coherence": 0.1, "groundedness": 0.5},
    ]
    front = pareto_front(points)
    assert {"coherence": 0.2, "groundedness": 0.1} not in front
    assert {"coherence": 0.4, "groundedness": 0.3} in front
    assert {"coherence": 0.1, "groundedness": 0.5} in front


def test_bayes_optimizer_proposes_and_tracks_history() -> None:
    specs = default_param_specs()
    state: dict[str, object] = {}
    rng = random.Random(17)
    optimizer = BayesParetoOptimizer(param_specs=specs, state=state, rng=rng)

    current_overlay = {"competition_top_k": 2, "competition_min_score": 0.2}
    proposal = optimizer.propose(current_overlay)
    assert isinstance(proposal.get("overlay"), dict)
    assert "acquisition" in proposal

    optimizer.observe_result(
        accepted=True,
        score=0.42,
        overlay=proposal["overlay"],  # type: ignore[index]
        objectives={"coherence": 0.2, "groundedness": 0.1},
        report={"trial_id": "trial-1"},
    )
    optimizer.observe_result(
        accepted=False,
        score=0.35,
        overlay=current_overlay,
        objectives={"coherence": 0.1, "groundedness": 0.05},
        report={"trial_id": "trial-2"},
    )

    assert state.get("accepted") == 1
    history = state.get("history")
    assert isinstance(history, list)
    assert len(history) == 2
    assert history[0]["trial_id"] == "trial-1"

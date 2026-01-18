import abc
from typing import Any, Dict
from ray.util.annotations import PublicAPI
@PublicAPI
class CombinedStopper(Stopper):
    """Combine several stoppers via 'OR'.

    Args:
        *stoppers: Stoppers to be combined.

    Examples:

        >>> import numpy as np
        >>> from ray import train, tune
        >>> from ray.tune.stopper import (
        ...     CombinedStopper,
        ...     MaximumIterationStopper,
        ...     TrialPlateauStopper,
        ... )
        >>>
        >>> stopper = CombinedStopper(
        ...     MaximumIterationStopper(max_iter=10),
        ...     TrialPlateauStopper(metric="my_metric"),
        ... )
        >>> def train_fn(config):
        ...     for i in range(15):
        ...         train.report({"my_metric": np.random.normal(0, 1 - i / 15)})
        ...
        >>> tuner = tune.Tuner(
        ...     train_fn,
        ...     run_config=train.RunConfig(stop=stopper),
        ... )
        >>> print("[ignore]"); result_grid = tuner.fit()  # doctest: +ELLIPSIS
        [ignore]...
        >>> all(result.metrics["training_iteration"] <= 20 for result in result_grid)
        True

    """

    def __init__(self, *stoppers: Stopper):
        self._stoppers = stoppers

    def __call__(self, trial_id: str, result: Dict[str, Any]) -> bool:
        return any((s(trial_id, result) for s in self._stoppers))

    def stop_all(self) -> bool:
        return any((s.stop_all() for s in self._stoppers))
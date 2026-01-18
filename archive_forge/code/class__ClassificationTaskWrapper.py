from typing import Any
from torchmetrics.metric import Metric
class _ClassificationTaskWrapper(Metric):
    """Base class for wrapper metrics for classification tasks."""

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update metric state."""
        raise NotImplementedError(f'{self.__class__.__name__} metric does not have a global `update` method. Use the task specific metric.')

    def compute(self) -> None:
        """Compute metric."""
        raise NotImplementedError(f'{self.__class__.__name__} metric does not have a global `compute` method. Use the task specific metric.')
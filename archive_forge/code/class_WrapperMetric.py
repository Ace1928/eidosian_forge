from typing import Any, Callable
from torchmetrics.metric import Metric
class WrapperMetric(Metric):
    """Abstract base class for wrapper metrics.

    Wrapper metrics are characterized by them wrapping another metric, and forwarding all calls to the wrapped metric.
    This means that all logic regarding synchronization etc. is handled by the wrapped metric, and the wrapper metric
    should not do anything in this regard.

    This class therefore overwrites all methods that are related to synchronization, and does nothing in them.

    Additionally, the forward method is not implemented by default as custom logic is required for each wrapper metric.

    """

    def _wrap_update(self, update: Callable) -> Callable:
        """Overwrite to do nothing, because the default wrapped functionality is handled by the wrapped metric."""
        return update

    def _wrap_compute(self, compute: Callable) -> Callable:
        """Overwrite to do nothing, because the default wrapped functionality is handled by the wrapped metric."""
        return compute

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Overwrite to do nothing, because the default wrapped functionality is handled by the wrapped metric."""
        raise NotImplementedError
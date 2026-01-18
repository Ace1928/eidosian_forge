from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, TypeVar, Union
import torch
from typing_extensions import override
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
class ThroughputMonitor(Throughput):
    """Computes throughput.

    This class will automatically keep a count of the number of log calls (``step``). But that can be modified as
    desired. For manual logging, using :class:`Throughput` directly might be desired.

    Example::

        logger = ...
        fabric = Fabric(logger=logger)
        throughput = ThroughputMonitor()
        t0 = time()
        for i in range(1, 100):
            do_work()
            if torch.cuda.is_available(): torch.cuda.synchronize()  # required or else time() won't be correct
            throughput.update(time=time() - t0, batches=i, samples=i)
            if i % 10 == 0:
                throughput.compute_and_log(step=i)

    Args:
        fabric: The Fabric object.
        \\**kwargs: See available parameters in :class:`Throughput`

    """

    def __init__(self, fabric: 'Fabric', **kwargs: Any) -> None:
        fabric._validate_launched()
        dtype = _plugin_to_compute_dtype(fabric.strategy.precision)
        available_flops = get_available_flops(fabric.device, dtype)
        super().__init__(available_flops=available_flops, world_size=fabric.world_size, **kwargs)
        self._fabric = fabric
        self.step = -1
        self.update = rank_zero_only(self.update)
        self.compute = rank_zero_only(self.compute, default={})
        self.compute_and_log = rank_zero_only(self.compute_and_log, default={})
        self.reset = rank_zero_only(self.reset)

    def compute_and_log(self, step: Optional[int]=None, **kwargs: Any) -> _THROUGHPUT_METRICS:
        """See :meth:`Throughput.compute`

        Args:
            step: Can be used to override the logging step.
            \\**kwargs: See available parameters in :meth:`Throughput.compute`

        """
        self.step = self.step + 1 if step is None else step
        metrics = self.compute(**kwargs)
        self._fabric.log_dict(metrics=metrics, step=self.step)
        return metrics
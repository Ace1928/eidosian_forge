import queue
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
import torch.multiprocessing as mp
from typing_extensions import override
from lightning_fabric.accelerators.xla import _XLA_AVAILABLE
from lightning_fabric.strategies.launchers.launcher import _Launcher
from lightning_fabric.strategies.launchers.multiprocessing import _GlobalStateSnapshot
from lightning_fabric.utilities.apply_func import move_data_to_device
def _wrapping_function(self, process_idx: int, function: Callable, args: Any, kwargs: Any, return_queue: Union[mp.SimpleQueue, queue.Queue], global_states: Optional[_GlobalStateSnapshot]=None) -> None:
    import torch_xla.core.xla_model as xm
    if len(xm.get_xla_supported_devices()) > 1:
        import copy
        function, args, kwargs = copy.deepcopy((function, args, kwargs))
    results = function(*args, **kwargs)
    if self._strategy.local_rank == 0:
        return_queue.put(move_data_to_device(results, 'cpu'))
    _rank_teardown(self._strategy.local_rank)
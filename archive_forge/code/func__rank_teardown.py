import queue
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
import torch.multiprocessing as mp
from typing_extensions import override
from lightning_fabric.accelerators.xla import _XLA_AVAILABLE
from lightning_fabric.strategies.launchers.launcher import _Launcher
from lightning_fabric.strategies.launchers.multiprocessing import _GlobalStateSnapshot
from lightning_fabric.utilities.apply_func import move_data_to_device
def _rank_teardown(rank: int) -> None:
    import torch_xla.core.xla_model as xm
    xm.rendezvous('end-process')
    if rank == 0:
        time.sleep(1)
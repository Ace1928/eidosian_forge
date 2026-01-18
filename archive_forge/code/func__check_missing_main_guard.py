import itertools
import os
from dataclasses import dataclass
from multiprocessing.queues import SimpleQueue
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional
import torch
import torch.backends.cudnn
import torch.multiprocessing as mp
from lightning_utilities import apply_to_collection
from torch.nn import Module
from typing_extensions import override
from lightning_fabric.accelerators.cpu import CPUAccelerator
from lightning_fabric.strategies.launchers.launcher import _Launcher
from lightning_fabric.utilities.apply_func import move_data_to_device
from lightning_fabric.utilities.distributed import _set_num_threads_if_needed
from lightning_fabric.utilities.imports import _IS_INTERACTIVE
from lightning_fabric.utilities.seed import _collect_rng_states, _set_rng_states
def _check_missing_main_guard() -> None:
    """Raises an exception if the ``__name__ == "__main__"`` guard is missing."""
    if not getattr(mp.current_process(), '_inheriting', False):
        return
    message = dedent('\n        Launching multiple processes with the \'spawn\' start method requires that your script guards the main\n        function with an `if __name__ == "__main__"` clause. For example:\n\n        def main():\n            # Put your code here\n            ...\n\n        if __name__ == "__main__":\n            main()\n\n        Alternatively, you can run with `strategy="ddp"` to avoid this error.\n        ')
    raise RuntimeError(message)
import os
import sys
import time
import traceback
from math import sqrt
from typing import Any, Callable, Dict, List, Optional, Sequence
from sphinx.errors import SphinxParallelError
from sphinx.util import logging
class SerialTasks:
    """Has the same interface as ParallelTasks, but executes tasks directly."""

    def __init__(self, nproc: int=1) -> None:
        pass

    def add_task(self, task_func: Callable, arg: Any=None, result_func: Optional[Callable]=None) -> None:
        if arg is not None:
            res = task_func(arg)
        else:
            res = task_func()
        if result_func:
            result_func(res)

    def join(self) -> None:
        pass
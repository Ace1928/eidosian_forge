import inspect
import logging
from queue import Queue
from functools import wraps
from typing import Callable, Dict, List
import torch.nn as nn
from torch.fx.graph_module import GraphModule
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult
def add_checks(self, check: Callable) -> None:
    """
        Adds a function which takes runs various checks on a given graph module.
        This function is run before and after each pass if the
        `run_checks_after_each_pass` flag is enabled.
        """
    sig = inspect.signature(check)
    if len(list(sig.parameters.values())) != 1:
        raise TypeError('PassManager check function should only take in one variable, a module')
    setattr(self, 'check', check)
import copy
import functools
import inspect
import os
import types
import warnings
from typing import Callable, Tuple
import pennylane as qml
from pennylane.typing import ResultBatch
def _device_wrapper(self, *targs, **tkwargs):

    def _wrapper(dev):
        new_dev = copy.deepcopy(dev)
        new_dev.batch_transform = lambda tape: self.construct(tape, *targs, **tkwargs)
        return new_dev
    return _wrapper
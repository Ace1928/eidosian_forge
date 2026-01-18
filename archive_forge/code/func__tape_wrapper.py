import copy
import functools
import inspect
import os
import types
import warnings
from typing import Callable, Tuple
import pennylane as qml
from pennylane.typing import ResultBatch
def _tape_wrapper(self, *targs, **tkwargs):
    return lambda tape: self.construct(tape, *targs, **tkwargs)
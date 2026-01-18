import copy
import functools
import inspect
import os
import types
import warnings
from typing import Callable, Tuple
import pennylane as qml
from pennylane.typing import ResultBatch
Applies a batch of post-processing functions to results.

        Args:
            res (ResultBatch): the results of executing a batch of circuits

        Returns:
            ResultBatch : results that have undergone classical post processing

        Closure variables:
            tape_counts: the number of tapes outputted from each application of the transform
            batch_fns: the post processing functions to apply to each sub-batch

        
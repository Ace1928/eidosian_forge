import functools
import os
import copy
import warnings
import types
from typing import Sequence
import pennylane as qml
from pennylane.typing import ResultBatch
def _batch_transform(self, original_batch, targs, tkwargs):
    """Apply the transform on a batch of tapes"""
    execution_tapes = []
    batch_fns = []
    tape_counts = []
    for t in original_batch:
        new_tapes, fn = self(t, *targs, **tkwargs)
        execution_tapes.extend(new_tapes)
        batch_fns.append(fn)
        tape_counts.append(len(new_tapes))

    def processing_fn(res: ResultBatch) -> ResultBatch:
        """Applies a batch of post-processing functions to results.

            Args:
                res (ResultBatch): the results of executing a batch of circuits

            Returns:
                ResultBatch : results that have undergone classical post processing

            Closure variables:
                tape_counts: the number of tapes outputted from each application of the transform
                batch_fns: the post processing functions to apply to each sub-batch

            """
        count = 0
        final_results = []
        for f, s in zip(batch_fns, tape_counts):
            new_res = f(res[count:count + s])
            final_results.append(new_res)
            count += s
        return tuple(final_results)
    return (tuple(execution_tapes), processing_fn)
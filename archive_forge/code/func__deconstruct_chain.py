from enum import Enum
from itertools import islice
from typing import (
import pandas
import ray
from ray._private.services import get_node_ip_address
from ray.util.client.common import ClientObjectRef
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.logging import get_logger
@classmethod
def _deconstruct_chain(cls, de: 'DeferredExecution', output: List, stack: List, result_consumers: List['DeferredExecution']):
    """
        Deconstruct the specified DeferredExecution chain.

        Parameters
        ----------
        de : DeferredExecution
            The chain to be deconstructed.
        output : list
            Put the arguments to this list.
        stack : list
            Used to eliminate recursive calls, that may lead to the RecursionError.
        result_consumers : list of DeferredExecution
            The result consumers.

        Yields
        ------
        Generator
            The ``_deconstruct_list()`` generator.
        """
    out_append = output.append
    out_extend = output.extend
    while True:
        de.unsubscribe()
        if (out_pos := getattr(de, 'out_pos', None)) and (not de.has_result):
            out_append(_Tag.REF)
            out_append(out_pos)
            output[out_pos] = out_pos
            if de.subscribers == 0:
                output[out_pos + 1] = 0
                result_consumers.remove(de)
            break
        elif not isinstance((data := de.data), DeferredExecution):
            if isinstance(data, ListOrTuple):
                yield cls._deconstruct_list(data, output, stack, result_consumers, out_append)
            else:
                out_append(data)
            if not de.has_result:
                stack.append(de)
            break
        else:
            stack.append(de)
            de = data
    while stack and isinstance(stack[-1], DeferredExecution):
        de: DeferredExecution = stack.pop()
        args = de.args
        kwargs = de.kwargs
        out_append(de.func)
        if de.flat_args:
            out_append(len(args))
            out_extend(args)
        else:
            out_append(-1)
            yield cls._deconstruct_list(args, output, stack, result_consumers, out_append)
        if de.flat_kwargs:
            out_append(len(kwargs))
            for item in kwargs.items():
                out_extend(item)
        else:
            out_append(-1)
            yield cls._deconstruct_list(kwargs.values(), output, stack, result_consumers, out_append)
            out_extend(kwargs)
        out_append(0)
        if de.subscribers > 0:
            de.out_pos = len(output) - 1
            result_consumers.append(de)
            out_append(de.num_returns)
        else:
            out_append(0)
from enum import Enum
from itertools import islice
from typing import (
import pandas
import ray
from ray._private.services import get_node_ip_address
from ray.util.client.common import ClientObjectRef
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.logging import get_logger
def _deconstruct(self) -> Tuple[List['DeferredExecution'], List[Any]]:
    """
        Convert the specified execution tree to a flat list.

        This is required for the automatic Ray object references
        materialization before passing the list to a Ray worker.

        The format of the list is the following:
        <input object> sequence<<function> <n><args> <n><kwargs> <ref> <nret>>...
        If <n> before <args> is >= 0, then the next n objects are the function arguments.
        If it is -1, it means that the method arguments contain list and/or
        DeferredExecution (chain) objects. In this case the next values are read
        one by one until `_Tag.END` is encountered. If the value is `_Tag.LIST`,
        then the next sequence of values up to `_Tag.END` is converted to list.
        If the value is `_Tag.CHAIN`, then the next sequence of values up to
        `_Tag.END` has exactly the same format, as described here.
        If the value is `_Tag.REF`, then the next value is a reference id, i.e.
        the actual value should be retrieved by this id from the previously
        saved objects. The <input object> could also be `_Tag.REF` or `_Tag.LIST`.

        If <n> before <kwargs> is >=0, then the next 2*n values are the argument
        names and values in the following format - [name1, value1, name2, value2...].
        If it's -1, then the next values are converted to list in the same way as
        <args> and the argument names are the next len(<args>) values.

        <ref> is an integer reference id. If it's not 0, then there is another
        chain referring to the execution result of this method and, thus, it must
        be saved so that other chains could retrieve the object by the id.

        <nret> field contains either the `num_returns` value or 0. If it's 0, the
        execution result is not returned, but is just passed to the next task in the
        chain. If it's 1, the result is returned as is. Otherwise, it's expected that
        the result is iterable and the specified number of values is returned from
        the iterator. The values lengths and widths are added to the meta list.

        Returns
        -------
        tuple of list
            * The first list is the result consumers.
                If a DeferredExecution has multiple subscribers, the execution result
                should be returned and saved in order to avoid duplicate executions.
                These DeferredExecution tasks are added to this list and, after the
                execution, the results are passed to the ``_set_result()`` method of
                each task.
            * The second is a flat list of arguments that could be passed to the remote executor.
        """
    stack = []
    result_consumers = []
    output = []
    stack.append(self._deconstruct_chain(self, output, stack, result_consumers))
    while stack:
        try:
            gen = stack.pop()
            next_gen = next(gen)
            stack.append(gen)
            stack.append(next_gen)
        except StopIteration:
            pass
    return (result_consumers, output)
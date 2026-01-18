from typing import List
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.base_physical_operator import NAryOperator
from ray.data._internal.stats import StatsDict
def _add_input_inner(self, refs: RefBundle, input_index: int) -> None:
    assert not self.completed()
    assert 0 <= input_index <= len(self._input_dependencies), input_index
    if not self._preserve_order:
        self._output_buffer.append(refs)
    elif input_index == self._input_idx_to_output:
        self._output_buffer.append(refs)
    else:
        self._input_buffers[input_index].append(refs)
from typing import List, Optional
from ray.data._internal.execution.interfaces import (
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.stats import StatsDict
class NAryOperator(PhysicalOperator):
    """An operator that has multiple input dependencies and one output.

    This operator serves as the base for union, zip, etc.
    """

    def __init__(self, *input_ops: LogicalOperator):
        """Create a OneToOneOperator.
        Args:
            input_op: Operator generating input data for this op.
            name: The name of this operator.
        """
        input_names = ', '.join([op._name for op in input_ops])
        op_name = f'{self.__class__.__name__}({input_names})'
        super().__init__(op_name, list(input_ops), target_max_block_size=None)
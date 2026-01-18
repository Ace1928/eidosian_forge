import numpy as np
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@property
def _nested_row_partitions(self):
    """The row_partitions representing this shape."""
    return [RowPartition.from_row_splits(rs) for rs in self.nested_row_splits]
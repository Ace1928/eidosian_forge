import json
from typing import Any, Dict, List, Tuple
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.convert import to_size
from triad.utils.hash import to_uuid
from triad.utils.pyarrow import SchemaedDataPartitioner
from triad.utils.schema import safe_split_out_of_quote, unquote_name
class DatasetPartitionCursor:
    """The cursor pointing at the first item of each logical partition inside
    a physical partition.

    It's important to understand the concept of partition, please read
    |PartitionTutorial|

    :param physical_partition_no: physical partition number passed in by
      :class:`~fugue.execution.execution_engine.ExecutionEngine`
    """

    def __init__(self, physical_partition_no: int):
        self._physical_partition_no = physical_partition_no
        self._partition_no = 0
        self._slice_no = 0

    def set(self, item: Any, partition_no: int, slice_no: int) -> None:
        """reset the cursor to a row (which should be the first row of a
        new logical partition)

        :param item: an item of the dataset, or an function generating the item
        :param partition_no: logical partition number
        :param slice_no: slice number inside the logical partition (to be deprecated)
        """
        self._item = item
        self._partition_no = partition_no
        self._slice_no = slice_no

    @property
    def item(self) -> Any:
        """Get current item"""
        if callable(self._item):
            self._item = self._item()
        return self._item

    @property
    def partition_no(self) -> int:
        """Logical partition number"""
        return self._partition_no

    @property
    def physical_partition_no(self) -> int:
        """Physical partition number"""
        return self._physical_partition_no

    @property
    def slice_no(self) -> int:
        """Slice number (inside the current logical partition), for now
        it should always be 0
        """
        return self._slice_no
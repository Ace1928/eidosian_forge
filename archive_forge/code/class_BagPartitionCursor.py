import json
from typing import Any, Dict, List, Tuple
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.convert import to_size
from triad.utils.hash import to_uuid
from triad.utils.pyarrow import SchemaedDataPartitioner
from triad.utils.schema import safe_split_out_of_quote, unquote_name
class BagPartitionCursor(DatasetPartitionCursor):
    """The cursor pointing at the first bag item of each logical partition inside
    a physical partition.

    It's important to understand the concept of partition, please read
    |PartitionTutorial|

    :param physical_partition_no: physical partition number passed in by
      :class:`~fugue.execution.execution_engine.ExecutionEngine`
    """
    pass
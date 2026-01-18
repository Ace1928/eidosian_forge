from typing import NamedTuple
from google.cloud.pubsublite.internal import fast_serialize
from google.cloud.pubsublite_v1.types.common import Cursor
from google.cloud.pubsublite.types.partition import Partition
@staticmethod
def _encode_parts(partition: int, offset: int) -> str:
    return fast_serialize.dump([partition, offset])
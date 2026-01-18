from .api import Request, Response
from .types import (
class ListPartitionReassignmentsResponse_v0(Response):
    API_KEY = 46
    API_VERSION = 0
    SCHEMA = Schema(('throttle_time_ms', Int32), ('error_code', Int16), ('error_message', CompactString('utf-8')), ('topics', CompactArray(('name', CompactString('utf-8')), ('partitions', CompactArray(('partition_index', Int32), ('replicas', CompactArray(Int32)), ('adding_replicas', CompactArray(Int32)), ('removing_replicas', CompactArray(Int32)), ('tags', TaggedFields))), ('tags', TaggedFields))), ('tags', TaggedFields))
from .api import Request, Response
from .types import (
class CreatePartitionsRequest_v0(Request):
    API_KEY = 37
    API_VERSION = 0
    RESPONSE_TYPE = CreatePartitionsResponse_v0
    SCHEMA = Schema(('topic_partitions', Array(('topic', String('utf-8')), ('new_partitions', Schema(('count', Int32), ('assignment', Array(Array(Int32))))))), ('timeout', Int32), ('validate_only', Boolean))
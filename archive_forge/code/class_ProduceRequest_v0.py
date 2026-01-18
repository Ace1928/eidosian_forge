from .api import Request, Response
from .types import Int16, Int32, Int64, String, Array, Schema, Bytes
class ProduceRequest_v0(ProduceRequest):
    API_VERSION = 0
    RESPONSE_TYPE = ProduceResponse_v0
    SCHEMA = Schema(('required_acks', Int16), ('timeout', Int32), ('topics', Array(('topic', String('utf-8')), ('partitions', Array(('partition', Int32), ('messages', Bytes))))))
from .api import Request, Response
from .types import Int16, Int32, Int64, String, Array, Schema, Bytes
class ProduceRequest_v3(ProduceRequest):
    API_VERSION = 3
    RESPONSE_TYPE = ProduceResponse_v3
    SCHEMA = Schema(('transactional_id', String('utf-8')), ('required_acks', Int16), ('timeout', Int32), ('topics', Array(('topic', String('utf-8')), ('partitions', Array(('partition', Int32), ('messages', Bytes))))))
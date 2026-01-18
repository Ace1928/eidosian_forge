from .api import Request, Response
from .types import Array, Boolean, Int16, Int32, Schema, String
class MetadataResponse_v1(Response):
    API_KEY = 3
    API_VERSION = 1
    SCHEMA = Schema(('brokers', Array(('node_id', Int32), ('host', String('utf-8')), ('port', Int32), ('rack', String('utf-8')))), ('controller_id', Int32), ('topics', Array(('error_code', Int16), ('topic', String('utf-8')), ('is_internal', Boolean), ('partitions', Array(('error_code', Int16), ('partition', Int32), ('leader', Int32), ('replicas', Array(Int32)), ('isr', Array(Int32)))))))
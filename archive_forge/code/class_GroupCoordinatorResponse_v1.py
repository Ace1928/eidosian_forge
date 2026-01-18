from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class GroupCoordinatorResponse_v1(Response):
    API_KEY = 10
    API_VERSION = 1
    SCHEMA = Schema(('error_code', Int16), ('error_message', String('utf-8')), ('coordinator_id', Int32), ('host', String('utf-8')), ('port', Int32))
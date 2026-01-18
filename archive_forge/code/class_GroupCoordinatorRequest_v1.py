from .api import Request, Response
from .types import Array, Int8, Int16, Int32, Int64, Schema, String
class GroupCoordinatorRequest_v1(Request):
    API_KEY = 10
    API_VERSION = 1
    RESPONSE_TYPE = GroupCoordinatorResponse_v1
    SCHEMA = Schema(('coordinator_key', String('utf-8')), ('coordinator_type', Int8))
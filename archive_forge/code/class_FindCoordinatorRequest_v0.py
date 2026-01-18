from .api import Request, Response
from .types import Int8, Int16, Int32, Schema, String
class FindCoordinatorRequest_v0(Request):
    API_KEY = 10
    API_VERSION = 0
    RESPONSE_TYPE = FindCoordinatorResponse_v0
    SCHEMA = Schema(('consumer_group', String('utf-8')))
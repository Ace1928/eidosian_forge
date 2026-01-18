from .api import Request, Response
from .struct import Struct
from .types import Array, Bytes, Int16, Int32, Schema, String
class ProtocolMetadata(Struct):
    SCHEMA = Schema(('version', Int16), ('subscription', Array(String('utf-8'))), ('user_data', Bytes))
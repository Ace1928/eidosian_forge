from aiokafka.protocol.struct import Struct
from aiokafka.protocol.types import Array, Bytes, Int16, Int32, Schema, String
from aiokafka.structs import TopicPartition
class ConsumerProtocolMemberMetadata(Struct):
    SCHEMA = Schema(('version', Int16), ('subscription', Array(String('utf-8'))), ('user_data', Bytes))
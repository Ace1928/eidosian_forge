from aiokafka.protocol.struct import Struct
from aiokafka.protocol.types import Array, Bytes, Int16, Int32, Schema, String
from aiokafka.structs import TopicPartition
class ConsumerProtocol(object):
    PROTOCOL_TYPE = 'consumer'
    ASSIGNMENT_STRATEGIES = ('range', 'roundrobin')
    METADATA = ConsumerProtocolMemberMetadata
    ASSIGNMENT = ConsumerProtocolMemberAssignment
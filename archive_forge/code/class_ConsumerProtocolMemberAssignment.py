from aiokafka.protocol.struct import Struct
from aiokafka.protocol.types import Array, Bytes, Int16, Int32, Schema, String
from aiokafka.structs import TopicPartition
class ConsumerProtocolMemberAssignment(Struct):
    SCHEMA = Schema(('version', Int16), ('assignment', Array(('topic', String('utf-8')), ('partitions', Array(Int32)))), ('user_data', Bytes))

    def partitions(self):
        return [TopicPartition(topic, partition) for topic, partitions in self.assignment for partition in partitions]
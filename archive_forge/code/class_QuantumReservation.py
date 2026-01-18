import proto
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
class QuantumReservation(proto.Message):
    """-

    Attributes:
        name (str):
            -
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        cancelled_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        whitelisted_users (Sequence[str]):
            -
    """
    name = proto.Field(proto.STRING, number=1)
    start_time = proto.Field(proto.MESSAGE, number=2, message=timestamp_pb2.Timestamp)
    end_time = proto.Field(proto.MESSAGE, number=3, message=timestamp_pb2.Timestamp)
    cancelled_time = proto.Field(proto.MESSAGE, number=4, message=timestamp_pb2.Timestamp)
    whitelisted_users = proto.RepeatedField(proto.STRING, number=5)
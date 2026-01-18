import proto
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
class QuantumReservationGrant(proto.Message):
    """-

    Attributes:
        name (str):
            -
        processor_names (Sequence[str]):
            -
        effective_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        expire_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        granted_duration (google.protobuf.duration_pb2.Duration):
            -
        available_duration (google.protobuf.duration_pb2.Duration):
            -
        budgets (Sequence[google.cloud.quantum_v1alpha1.types.QuantumReservationGrant.Budget]):
            -
    """

    class Budget(proto.Message):
        """-

        Attributes:
            project_id (str):
                -
            granted_duration (google.protobuf.duration_pb2.Duration):
                -
            available_duration (google.protobuf.duration_pb2.Duration):
                -
        """
        project_id = proto.Field(proto.STRING, number=1)
        granted_duration = proto.Field(proto.MESSAGE, number=2, message=duration_pb2.Duration)
        available_duration = proto.Field(proto.MESSAGE, number=3, message=duration_pb2.Duration)
    name = proto.Field(proto.STRING, number=1)
    processor_names = proto.RepeatedField(proto.STRING, number=2)
    effective_time = proto.Field(proto.MESSAGE, number=3, message=timestamp_pb2.Timestamp)
    expire_time = proto.Field(proto.MESSAGE, number=4, message=timestamp_pb2.Timestamp)
    granted_duration = proto.Field(proto.MESSAGE, number=5, message=duration_pb2.Duration)
    available_duration = proto.Field(proto.MESSAGE, number=6, message=duration_pb2.Duration)
    budgets = proto.RepeatedField(proto.MESSAGE, number=7, message=Budget)
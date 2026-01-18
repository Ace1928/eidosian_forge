import proto
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
class ReallocateQuantumReservationGrantRequest(proto.Message):
    """-

    Attributes:
        name (str):
            -
        source_project_id (str):
            -
        target_project_id (str):
            -
        duration (google.protobuf.duration_pb2.Duration):
            -
    """
    name = proto.Field(proto.STRING, number=1)
    source_project_id = proto.Field(proto.STRING, number=2)
    target_project_id = proto.Field(proto.STRING, number=3)
    duration = proto.Field(proto.MESSAGE, number=4, message=duration_pb2.Duration)
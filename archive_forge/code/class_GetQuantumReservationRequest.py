import proto
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
class GetQuantumReservationRequest(proto.Message):
    """-

    Attributes:
        name (str):
            -
    """
    name = proto.Field(proto.STRING, number=1)
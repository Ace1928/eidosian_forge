import proto
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
class GetQuantumJobRequest(proto.Message):
    """-

    Attributes:
        name (str):
            -
        return_run_context (bool):
            -
    """
    name = proto.Field(proto.STRING, number=1)
    return_run_context = proto.Field(proto.BOOL, number=2)
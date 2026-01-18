import proto
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
class ListQuantumTimeSlotsRequest(proto.Message):
    """-

    Attributes:
        parent (str):
            -
        page_size (int):
            -
        page_token (str):
            -
        filter (str):
            -
    """
    parent = proto.Field(proto.STRING, number=1)
    page_size = proto.Field(proto.INT32, number=2)
    page_token = proto.Field(proto.STRING, number=3)
    filter = proto.Field(proto.STRING, number=4)
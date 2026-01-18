import proto
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
class UpdateQuantumReservationRequest(proto.Message):
    """-

    Attributes:
        name (str):
            -
        quantum_reservation (google.cloud.quantum_v1alpha1.types.QuantumReservation):
            -
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            -
    """
    name = proto.Field(proto.STRING, number=1)
    quantum_reservation = proto.Field(proto.MESSAGE, number=2, message=quantum.QuantumReservation)
    update_mask = proto.Field(proto.MESSAGE, number=3, message=field_mask_pb2.FieldMask)
import proto
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
class ListQuantumCalibrationsRequest(proto.Message):
    """-

    Attributes:
        parent (str):
            -
        view (google.cloud.quantum_v1alpha1.types.ListQuantumCalibrationsRequest.QuantumCalibrationView):
            -
        page_size (int):
            -
        page_token (str):
            -
        filter (str):
            -
    """

    class QuantumCalibrationView(proto.Enum):
        """-"""
        QUANTUM_CALIBRATION_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    parent = proto.Field(proto.STRING, number=1)
    view = proto.Field(proto.ENUM, number=5, enum=QuantumCalibrationView)
    page_size = proto.Field(proto.INT32, number=2)
    page_token = proto.Field(proto.STRING, number=3)
    filter = proto.Field(proto.STRING, number=4)
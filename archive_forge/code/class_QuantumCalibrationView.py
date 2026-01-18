import proto
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
class QuantumCalibrationView(proto.Enum):
    """-"""
    QUANTUM_CALIBRATION_VIEW_UNSPECIFIED = 0
    BASIC = 1
    FULL = 2
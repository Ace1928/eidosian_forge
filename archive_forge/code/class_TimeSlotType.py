import proto
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
class TimeSlotType(proto.Enum):
    """-"""
    TIME_SLOT_TYPE_UNSPECIFIED = 0
    MAINTENANCE = 1
    OPEN_SWIM = 2
    RESERVATION = 3
    UNALLOCATED = 4
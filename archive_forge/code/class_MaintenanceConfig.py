import proto
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
class MaintenanceConfig(proto.Message):
    """-

        Attributes:
            title (str):
                -
            description (str):
                -
        """
    title = proto.Field(proto.STRING, number=1)
    description = proto.Field(proto.STRING, number=2)
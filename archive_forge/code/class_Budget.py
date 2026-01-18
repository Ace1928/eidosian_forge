import proto
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
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
import proto
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
class GcsLocation(proto.Message):
    """-

    Attributes:
        uri (str):
            -
        type_url (str):
            -
    """
    uri = proto.Field(proto.STRING, number=1)
    type_url = proto.Field(proto.STRING, number=2)
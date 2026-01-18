import enum
from cloudsdk.google.protobuf import descriptor_pb2
from proto import _file_info
from proto import _package_info
from proto.marshal.rules.enums import EnumRule
class _EnumInfo:

    def __init__(self, *, full_name: str, pb):
        self.full_name = full_name
        self.pb = pb
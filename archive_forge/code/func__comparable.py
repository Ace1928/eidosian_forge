import enum
from cloudsdk.google.protobuf import descriptor_pb2
from proto import _file_info
from proto import _package_info
from proto.marshal.rules.enums import EnumRule
def _comparable(self, other):
    return type(other) in (type(self), int)
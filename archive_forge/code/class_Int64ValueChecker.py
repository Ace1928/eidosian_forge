import ctypes
import numbers
from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf.internal import decoder
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import descriptor
class Int64ValueChecker(IntValueChecker):
    _MIN = -(1 << 63)
    _MAX = (1 << 63) - 1
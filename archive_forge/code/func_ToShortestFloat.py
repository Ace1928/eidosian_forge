import ctypes
import numbers
from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf.internal import decoder
from cloudsdk.google.protobuf.internal import encoder
from cloudsdk.google.protobuf.internal import wire_format
from cloudsdk.google.protobuf import descriptor
def ToShortestFloat(original):
    """Returns the shortest float that has same value in wire."""
    precision = 6
    rounded = float('{0:.{1}g}'.format(original, precision))
    while TruncateToFourByteFloat(rounded) != original:
        precision += 1
        rounded = float('{0:.{1}g}'.format(original, precision))
    return rounded
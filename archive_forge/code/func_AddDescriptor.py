import collections
import warnings
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import descriptor_database
from cloudsdk.google.protobuf import text_encoding
@_Deprecated
def AddDescriptor(self, desc):
    self._AddDescriptor(desc)
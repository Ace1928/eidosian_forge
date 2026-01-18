import collections
import warnings
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import descriptor_database
from cloudsdk.google.protobuf import text_encoding
@_Deprecated
def AddFileDescriptor(self, file_desc):
    self._InternalAddFileDescriptor(file_desc)
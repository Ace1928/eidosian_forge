import collections
import warnings
from cloudsdk.google.protobuf import descriptor
from cloudsdk.google.protobuf import descriptor_database
from cloudsdk.google.protobuf import text_encoding
@_Deprecated
def AddExtensionDescriptor(self, extension):
    self._AddExtensionDescriptor(extension)
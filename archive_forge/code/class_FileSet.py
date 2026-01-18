import codecs
import types
import six
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
class FileSet(messages.Message):
    """A collection of FileDescriptors.

    Fields:
      files: Files in file-set.
    """
    files = messages.MessageField(FileDescriptor, 1, repeated=True)
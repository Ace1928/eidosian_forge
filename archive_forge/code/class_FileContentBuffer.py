from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileContentBuffer(_messages.Message):
    """A FileContentBuffer object.

  Enums:
    FileTypeValueValuesEnum: The file type of source file.

  Fields:
    content: The raw content in the secure keys file.
    fileType: The file type of source file.
  """

    class FileTypeValueValuesEnum(_messages.Enum):
        """The file type of source file.

    Values:
      BIN: <no description>
      UNDEFINED: <no description>
      X509: <no description>
    """
        BIN = 0
        UNDEFINED = 1
        X509 = 2
    content = _messages.BytesField(1)
    fileType = _messages.EnumField('FileTypeValueValuesEnum', 2)
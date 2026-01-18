from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SDKInfo(_messages.Message):
    """SDK Information.

  Enums:
    LanguageValueValuesEnum: Required. The SDK Language.

  Fields:
    language: Required. The SDK Language.
    version: Optional. The SDK version.
  """

    class LanguageValueValuesEnum(_messages.Enum):
        """Required. The SDK Language.

    Values:
      UNKNOWN: UNKNOWN Language.
      JAVA: Java.
      PYTHON: Python.
      GO: Go.
    """
        UNKNOWN = 0
        JAVA = 1
        PYTHON = 2
        GO = 3
    language = _messages.EnumField('LanguageValueValuesEnum', 1)
    version = _messages.StringField(2)
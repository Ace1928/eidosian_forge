from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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
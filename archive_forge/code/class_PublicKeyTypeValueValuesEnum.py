from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PublicKeyTypeValueValuesEnum(_messages.Enum):
    """The output format of the public key requested. X509_PEM is the default
    output format.

    Values:
      TYPE_NONE: <no description>
      TYPE_X509_PEM_FILE: <no description>
      TYPE_RAW_PUBLIC_KEY: <no description>
    """
    TYPE_NONE = 0
    TYPE_X509_PEM_FILE = 1
    TYPE_RAW_PUBLIC_KEY = 2
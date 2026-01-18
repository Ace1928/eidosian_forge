from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PortFieldTypeValueValuesEnum(_messages.Enum):
    """Whether port number should be provided by customers.

    Values:
      FIELD_TYPE_UNSPECIFIED: <no description>
      REQUIRED: <no description>
      OPTIONAL: <no description>
      NOT_USED: <no description>
    """
    FIELD_TYPE_UNSPECIFIED = 0
    REQUIRED = 1
    OPTIONAL = 2
    NOT_USED = 3
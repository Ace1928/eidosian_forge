from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ParentTypeValueValuesEnum(_messages.Enum):
    """The resource hierarchy level at which the data profile was generated.

    Values:
      PARENT_TYPE_UNSPECIFIED: Unspecified parent type.
      ORGANIZATION: Organization-level configurations.
      PROJECT: Project-level configurations.
    """
    PARENT_TYPE_UNSPECIFIED = 0
    ORGANIZATION = 1
    PROJECT = 2
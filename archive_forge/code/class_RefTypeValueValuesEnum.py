from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RefTypeValueValuesEnum(_messages.Enum):
    """Type of refs to fetch

    Values:
      REF_TYPE_UNSPECIFIED: No type specified.
      TAG: To fetch tags.
      BRANCH: To fetch branches.
    """
    REF_TYPE_UNSPECIFIED = 0
    TAG = 1
    BRANCH = 2
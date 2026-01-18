from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LifecycleValueValuesEnum(_messages.Enum):
    """Optional. lifecycle of the release

    Values:
      LIFECYCLE_UNSPECIFIED: indicates lifecycle has not been specified.
      DRAFT: indicates that release is being edited.
      PUBLISHED: indicates that release is now published (or released) and
        immutable.
    """
    LIFECYCLE_UNSPECIFIED = 0
    DRAFT = 1
    PUBLISHED = 2
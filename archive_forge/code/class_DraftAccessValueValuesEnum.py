from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DraftAccessValueValuesEnum(_messages.Enum):
    """Defines the level of data access when a compose time add-on is
    triggered.

    Values:
      UNSPECIFIED: Default value when nothing is set for draftAccess.
      NONE: The compose trigger can't access any data of the draft when a
        compose add-on is triggered.
      METADATA: Gives the compose trigger the permission to access the
        metadata of the draft when a compose add-on is triggered. This
        includes the audience list, such as the To and Cc list of a draft
        message.
    """
    UNSPECIFIED = 0
    NONE = 1
    METADATA = 2
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PusherValueValuesEnum(_messages.Enum):
    """Allowed PR role that triggers a Workflow.

    Values:
      PUSHER_UNSPECIFIED: Default to OWNER_AND_COLLABORATOR.
      OWNER_AND_COLLABORATORS: PR author are ownes and/or collaborators of the
        SCM repo.
      OWNER: PR author is the owner of the SCM repo.
      ALL_USERS: PR author can be everyone.
    """
    PUSHER_UNSPECIFIED = 0
    OWNER_AND_COLLABORATORS = 1
    OWNER = 2
    ALL_USERS = 3
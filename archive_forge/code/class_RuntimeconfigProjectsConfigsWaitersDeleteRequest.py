from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeconfigProjectsConfigsWaitersDeleteRequest(_messages.Message):
    """A RuntimeconfigProjectsConfigsWaitersDeleteRequest object.

  Fields:
    name: The Waiter resource to delete, in the format:
      `projects/[PROJECT_ID]/configs/[CONFIG_NAME]/waiters/[WAITER_NAME]`
  """
    name = _messages.StringField(1, required=True)
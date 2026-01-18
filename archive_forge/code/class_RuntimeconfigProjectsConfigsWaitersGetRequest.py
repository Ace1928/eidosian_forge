from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeconfigProjectsConfigsWaitersGetRequest(_messages.Message):
    """A RuntimeconfigProjectsConfigsWaitersGetRequest object.

  Fields:
    name: The fully-qualified name of the Waiter resource object to retrieve,
      in the format:
      `projects/[PROJECT_ID]/configs/[CONFIG_NAME]/waiters/[WAITER_NAME]`
  """
    name = _messages.StringField(1, required=True)
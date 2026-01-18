from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegisterInstanceRequest(_messages.Message):
    """Request for registering a notebook instance.

  Fields:
    instanceId: Required. User defined unique ID of this instance. The
      `instance_id` must be 1 to 63 characters long and contain only lowercase
      letters, numeric characters, and dashes. The first character must be a
      lowercase letter and the last character cannot be a dash.
  """
    instanceId = _messages.StringField(1)
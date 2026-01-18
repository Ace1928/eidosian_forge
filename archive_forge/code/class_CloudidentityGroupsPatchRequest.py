from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsPatchRequest(_messages.Message):
    """A CloudidentityGroupsPatchRequest object.

  Fields:
    group: A Group resource to be passed as the request body.
    name: Output only. The [resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      `Group`. Shall be of the form `groups/{group}`.
    updateMask: Required. The names of fields to update. May only contain the
      following field names: `display_name`, `description`, `labels`.
  """
    group = _messages.MessageField('Group', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
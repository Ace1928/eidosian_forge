from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsDeleteRequest(_messages.Message):
    """A CloudidentityGroupsDeleteRequest object.

  Fields:
    name: Required. The [resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      `Group` to retrieve. Must be of the form `groups/{group}`.
  """
    name = _messages.StringField(1, required=True)
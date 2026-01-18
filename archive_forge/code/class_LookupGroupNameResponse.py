from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookupGroupNameResponse(_messages.Message):
    """The response message for GroupsService.LookupGroupName.

  Fields:
    name: The [resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      looked-up `Group`.
  """
    name = _messages.StringField(1)
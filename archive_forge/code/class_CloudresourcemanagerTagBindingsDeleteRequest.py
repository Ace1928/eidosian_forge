from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerTagBindingsDeleteRequest(_messages.Message):
    """A CloudresourcemanagerTagBindingsDeleteRequest object.

  Fields:
    name: Required. The name of the TagBinding. This is a String of the form:
      `tagBindings/{id}` (e.g. `tagBindings/%2F%2Fcloudresourcemanager.googlea
      pis.com%2Fprojects%2F123/tagValues/456`).
  """
    name = _messages.StringField(1, required=True)
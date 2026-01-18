from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TagBinding(_messages.Message):
    """A TagBinding represents a connection between a TagValue and a cloud
  resource Once a TagBinding is created, the TagValue is applied to all the
  descendants of the Google Cloud resource.

  Fields:
    name: Output only. The name of the TagBinding. This is a String of the
      form: `tagBindings/{full-resource-name}/{tag-value-name}` (e.g. `tagBind
      ings/%2F%2Fcloudresourcemanager.googleapis.com%2Fprojects%2F123/tagValue
      s/456`).
    parent: The full resource name of the resource the TagValue is bound to.
      E.g. `//cloudresourcemanager.googleapis.com/projects/123`
    tagValue: The TagValue of the TagBinding. Must be of the form
      `tagValues/456`.
    tagValueNamespacedName: The namespaced name for the TagValue of the
      TagBinding. Must be in the format
      `{parent_id}/{tag_key_short_name}/{short_name}`. For methods that
      support TagValue namespaced name, only one of tag_value_namespaced_name
      or tag_value may be filled. Requests with both fields will be rejected.
  """
    name = _messages.StringField(1)
    parent = _messages.StringField(2)
    tagValue = _messages.StringField(3)
    tagValueNamespacedName = _messages.StringField(4)
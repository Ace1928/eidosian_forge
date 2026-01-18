from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TagValue(_messages.Message):
    """A TagValue is a child of a particular TagKey. This is used to group
  cloud resources for the purpose of controlling them using policies.

  Fields:
    createTime: Output only. Creation time.
    description: Optional. User-assigned description of the TagValue. Must not
      exceed 256 characters. Read-write.
    etag: Optional. Entity tag which users can pass to prevent race
      conditions. This field is always set in server responses. See
      UpdateTagValueRequest for details.
    name: Immutable. Resource name for TagValue in the format `tagValues/456`.
    namespacedName: Output only. The namespaced name of the TagValue. Can be
      in the form
      `{organization_id}/{tag_key_short_name}/{tag_value_short_name}` or
      `{project_id}/{tag_key_short_name}/{tag_value_short_name}` or
      `{project_number}/{tag_key_short_name}/{tag_value_short_name}`.
    parent: Immutable. The resource name of the new TagValue's parent TagKey.
      Must be of the form `tagKeys/{tag_key_id}`.
    shortName: Required. Immutable. User-assigned short name for TagValue. The
      short name should be unique for TagValues within the same parent TagKey.
      The short name must be 63 characters or less, beginning and ending with
      an alphanumeric character ([a-z0-9A-Z]) with dashes (-), underscores
      (_), dots (.), and alphanumerics between.
    updateTime: Output only. Update time.
  """
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    etag = _messages.StringField(3)
    name = _messages.StringField(4)
    namespacedName = _messages.StringField(5)
    parent = _messages.StringField(6)
    shortName = _messages.StringField(7)
    updateTime = _messages.StringField(8)
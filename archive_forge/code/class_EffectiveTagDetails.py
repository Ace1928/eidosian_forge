from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EffectiveTagDetails(_messages.Message):
    """The effective tags and the ancestor resources from which they were
  inherited.

  Fields:
    attachedResource: The [full resource name](https://cloud.google.com/asset-
      inventory/docs/resource-name-format) of the ancestor from which an
      effective_tag is inherited, according to [tag
      inheritance](https://cloud.google.com/resource-manager/docs/tags/tags-
      overview#inheritance).
    effectiveTags: The effective tags inherited from the attached_resource.
      Note that tags with the same key but different values may attach to
      resources at a different hierarchy levels. The lower hierarchy tag value
      will overwrite the higher hierarchy tag value of the same tag key. In
      this case, the tag value at the higher hierarchy level will be removed.
      For more information, see [tag
      inheritance](https://cloud.google.com/resource-manager/docs/tags/tags-
      overview#inheritance).
  """
    attachedResource = _messages.StringField(1)
    effectiveTags = _messages.MessageField('Tag', 2, repeated=True)
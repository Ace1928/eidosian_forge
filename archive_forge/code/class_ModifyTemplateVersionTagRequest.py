from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModifyTemplateVersionTagRequest(_messages.Message):
    """Add a tag to the current TemplateVersion. If tag exist in another
  TemplateVersion in the Template, remove the tag before add it to the current
  TemplateVersion. If remove_only set, remove the tag from the current
  TemplateVersion.

  Fields:
    removeOnly: The flag that indicates if the request is only for remove tag
      from TemplateVersion.
    tag: The tag for update.
  """
    removeOnly = _messages.BooleanField(1)
    tag = _messages.StringField(2)
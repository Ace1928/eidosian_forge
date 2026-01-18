from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1RenameTagTemplateFieldRequest(_messages.Message):
    """Request message for RenameTagTemplateField.

  Fields:
    newTagTemplateFieldId: Required. The new ID of this tag template field.
      For example, `my_new_field`.
  """
    newTagTemplateFieldId = _messages.StringField(1)
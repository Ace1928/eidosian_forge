from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesTemplateDeleteRequest(_messages.Message):
    """A FusiontablesTemplateDeleteRequest object.

  Fields:
    tableId: Table from which the template is being deleted
    templateId: Identifier for the template which is being deleted
  """
    tableId = _messages.StringField(1, required=True)
    templateId = _messages.IntegerField(2, required=True, variant=_messages.Variant.INT32)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesTableUpdateRequest(_messages.Message):
    """A FusiontablesTableUpdateRequest object.

  Fields:
    replaceViewDefinition: Should the view definition also be updated? The
      specified view definition replaces the existing one. Only a view can be
      updated with a new definition.
    table: A Table resource to be passed as the request body.
    tableId: ID of the table that is being updated.
  """
    replaceViewDefinition = _messages.BooleanField(1)
    table = _messages.MessageField('Table', 2)
    tableId = _messages.StringField(3, required=True)
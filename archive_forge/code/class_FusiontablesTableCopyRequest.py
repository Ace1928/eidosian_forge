from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesTableCopyRequest(_messages.Message):
    """A FusiontablesTableCopyRequest object.

  Fields:
    copyPresentation: Whether to also copy tabs, styles, and templates.
      Default is false.
    tableId: ID of the table that is being copied.
  """
    copyPresentation = _messages.BooleanField(1)
    tableId = _messages.StringField(2, required=True)
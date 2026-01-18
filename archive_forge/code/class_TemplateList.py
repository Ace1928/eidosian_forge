from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class TemplateList(_messages.Message):
    """Represents a list of templates for a given table.

  Fields:
    items: List of all requested templates.
    kind: Type name: a list of all templates.
    nextPageToken: Token used to access the next page of this result. No token
      is displayed if there are no more pages left.
    totalItems: Total number of templates for the table.
  """
    items = _messages.MessageField('Template', 1, repeated=True)
    kind = _messages.StringField(2, default=u'fusiontables#templateList')
    nextPageToken = _messages.StringField(3)
    totalItems = _messages.IntegerField(4, variant=_messages.Variant.INT32)
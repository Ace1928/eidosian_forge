from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProvidersNotesOccurrencesListRequest(_messages.Message):
    """A ContaineranalysisProvidersNotesOccurrencesListRequest object.

  Fields:
    filter: The filter expression.
    name: The name field will contain the note name for example:
      "provider/{provider_id}/notes/{note_id}"
    pageSize: Number of notes to return in the list.
    pageToken: Token to provide to skip to a particular spot in the list.
  """
    filter = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
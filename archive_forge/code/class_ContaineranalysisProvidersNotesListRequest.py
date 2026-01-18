from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProvidersNotesListRequest(_messages.Message):
    """A ContaineranalysisProvidersNotesListRequest object.

  Fields:
    filter: The filter expression.
    name: The name field will contain the project Id for example:
      "providers/{provider_id} @Deprecated
    pageSize: Number of notes to return in the list.
    pageToken: Token to provide to skip to a particular spot in the list.
    parent: This field contains the project Id for example:
      "projects/{PROJECT_ID}".
  """
    filter = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsNotesGetRequest(_messages.Message):
    """A ContaineranalysisProjectsNotesGetRequest object.

  Fields:
    name: Required. The name of the note in the form of
      `projects/[PROVIDER_ID]/notes/[NOTE_ID]`.
  """
    name = _messages.StringField(1, required=True)
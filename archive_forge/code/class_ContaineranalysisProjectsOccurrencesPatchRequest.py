from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsOccurrencesPatchRequest(_messages.Message):
    """A ContaineranalysisProjectsOccurrencesPatchRequest object.

  Fields:
    name: Required. The name of the occurrence in the form of
      `projects/[PROJECT_ID]/occurrences/[OCCURRENCE_ID]`.
    occurrence: A Occurrence resource to be passed as the request body.
    updateMask: The fields to update.
  """
    name = _messages.StringField(1, required=True)
    occurrence = _messages.MessageField('Occurrence', 2)
    updateMask = _messages.StringField(3)
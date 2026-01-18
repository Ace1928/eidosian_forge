from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsOccurrencesBatchCreateRequest(_messages.Message):
    """A ContaineranalysisProjectsOccurrencesBatchCreateRequest object.

  Fields:
    batchCreateOccurrencesRequest: A BatchCreateOccurrencesRequest resource to
      be passed as the request body.
    parent: Required. The name of the project in the form of
      `projects/[PROJECT_ID]`, under which the occurrences are to be created.
  """
    batchCreateOccurrencesRequest = _messages.MessageField('BatchCreateOccurrencesRequest', 1)
    parent = _messages.StringField(2, required=True)
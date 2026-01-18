from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastoreProjectsRunAggregationQueryRequest(_messages.Message):
    """A DatastoreProjectsRunAggregationQueryRequest object.

  Fields:
    projectId: Required. The ID of the project against which to make the
      request.
    runAggregationQueryRequest: A RunAggregationQueryRequest resource to be
      passed as the request body.
  """
    projectId = _messages.StringField(1, required=True)
    runAggregationQueryRequest = _messages.MessageField('RunAggregationQueryRequest', 2)
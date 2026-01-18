from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryJobsQueryRequest(_messages.Message):
    """A BigqueryJobsQueryRequest object.

  Fields:
    projectId: Project ID of the project billed for the query
    queryRequest: A QueryRequest resource to be passed as the request body.
  """
    projectId = _messages.StringField(1, required=True)
    queryRequest = _messages.MessageField('QueryRequest', 2)
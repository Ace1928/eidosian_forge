from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsHostQueriesGetResultRequest(_messages.Message):
    """A ApigeeOrganizationsHostQueriesGetResultRequest object.

  Fields:
    name: Required. Name of the asynchronous query result to get. Must be of
      the form `organizations/{org}/queries/{queryId}/result`.
  """
    name = _messages.StringField(1, required=True)
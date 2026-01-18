from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsQueriesGetResulturlRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsQueriesGetResulturlRequest object.

  Fields:
    name: Required. Name of the asynchronous query result to get. Must be of
      the form
      `organizations/{org}/environments/{env}/queries/{queryId}/resulturl`.
  """
    name = _messages.StringField(1, required=True)
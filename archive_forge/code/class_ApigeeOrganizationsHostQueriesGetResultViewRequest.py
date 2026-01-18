from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsHostQueriesGetResultViewRequest(_messages.Message):
    """A ApigeeOrganizationsHostQueriesGetResultViewRequest object.

  Fields:
    name: Required. Name of the asynchronous query result view to get. Must be
      of the form `organizations/{org}/queries/{queryId}/resultView`.
  """
    name = _messages.StringField(1, required=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSharedflowsRevisionsGetRequest(_messages.Message):
    """A ApigeeOrganizationsSharedflowsRevisionsGetRequest object.

  Fields:
    format: Specify `bundle` to export the contents of the shared flow bundle.
      Otherwise, the bundle metadata is returned.
    name: Required. The name of the shared flow revision to get. Must be of
      the form: `organizations/{organization_id}/sharedflows/{shared_flow_id}/
      revisions/{revision_id}`
  """
    format = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
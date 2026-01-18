from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSharedflowsRevisionsDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsSharedflowsRevisionsDeleteRequest object.

  Fields:
    name: Required. The name of the shared flow revision to delete. Must be of
      the form: `organizations/{organization_id}/sharedflows/{shared_flow_id}/
      revisions/{revision_id}`
  """
    name = _messages.StringField(1, required=True)
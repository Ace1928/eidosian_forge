from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSharedflowsRevisionsUpdateSharedFlowRevisionRequest(_messages.Message):
    """A ApigeeOrganizationsSharedflowsRevisionsUpdateSharedFlowRevisionRequest
  object.

  Fields:
    googleApiHttpBody: A GoogleApiHttpBody resource to be passed as the
      request body.
    name: Required. The name of the shared flow revision to update. Must be of
      the form: `organizations/{organization_id}/sharedflows/{shared_flow_id}/
      revisions/{revision_id}`
    validate: Ignored. All uploads are validated regardless of the value of
      this field. It is kept for compatibility with existing APIs. Must be
      `true` or `false` if provided.
  """
    googleApiHttpBody = _messages.MessageField('GoogleApiHttpBody', 1)
    name = _messages.StringField(2, required=True)
    validate = _messages.BooleanField(3)
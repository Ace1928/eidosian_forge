from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSharedflowsDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsSharedflowsDeleteRequest object.

  Fields:
    name: Required. shared flow name of the form:
      `organizations/{organization_id}/sharedflows/{shared_flow_id}`
  """
    name = _messages.StringField(1, required=True)
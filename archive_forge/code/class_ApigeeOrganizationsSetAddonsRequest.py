from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSetAddonsRequest(_messages.Message):
    """A ApigeeOrganizationsSetAddonsRequest object.

  Fields:
    googleCloudApigeeV1SetAddonsRequest: A GoogleCloudApigeeV1SetAddonsRequest
      resource to be passed as the request body.
    org: Required. Name of the organization. Use the following structure in
      your request: `organizations/{org}`
  """
    googleCloudApigeeV1SetAddonsRequest = _messages.MessageField('GoogleCloudApigeeV1SetAddonsRequest', 1)
    org = _messages.StringField(2, required=True)
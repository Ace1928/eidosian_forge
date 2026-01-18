from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsUndeleteRequest(_messages.Message):
    """A ApigeeOrganizationsUndeleteRequest object.

  Fields:
    googleCloudApigeeV1UndeleteOrganizationRequest: A
      GoogleCloudApigeeV1UndeleteOrganizationRequest resource to be passed as
      the request body.
    name: Required. Name of the organization. Use the following structure in
      your request: `organizations/{org}`
  """
    googleCloudApigeeV1UndeleteOrganizationRequest = _messages.MessageField('GoogleCloudApigeeV1UndeleteOrganizationRequest', 1)
    name = _messages.StringField(2, required=True)
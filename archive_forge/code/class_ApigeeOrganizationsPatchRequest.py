from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsPatchRequest(_messages.Message):
    """A ApigeeOrganizationsPatchRequest object.

  Fields:
    googleCloudApigeeV1Organization: A GoogleCloudApigeeV1Organization
      resource to be passed as the request body.
    name: Required. Apigee organization name in the following format:
      `organizations/{org}`
    updateMask: List of fields to be updated. Fields that can be updated:
      release_channel.
  """
    googleCloudApigeeV1Organization = _messages.MessageField('GoogleCloudApigeeV1Organization', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
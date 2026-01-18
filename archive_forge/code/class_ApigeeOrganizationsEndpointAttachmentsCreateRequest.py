from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEndpointAttachmentsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsEndpointAttachmentsCreateRequest object.

  Fields:
    endpointAttachmentId: ID to use for the endpoint attachment. ID must start
      with a lowercase letter followed by up to 31 lowercase letters, numbers,
      or hyphens, and cannot end with a hyphen. The minimum length is 2.
    googleCloudApigeeV1EndpointAttachment: A
      GoogleCloudApigeeV1EndpointAttachment resource to be passed as the
      request body.
    parent: Required. Organization the endpoint attachment will be created in.
  """
    endpointAttachmentId = _messages.StringField(1)
    googleCloudApigeeV1EndpointAttachment = _messages.MessageField('GoogleCloudApigeeV1EndpointAttachment', 2)
    parent = _messages.StringField(3, required=True)
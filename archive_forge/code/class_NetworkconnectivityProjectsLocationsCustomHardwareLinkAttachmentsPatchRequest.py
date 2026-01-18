from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkconnectivityProjectsLocationsCustomHardwareLinkAttachmentsPatchRequest(_messages.Message):
    """A NetworkconnectivityProjectsLocationsCustomHardwareLinkAttachmentsPatch
  Request object.

  Fields:
    googleCloudNetworkconnectivityV1betaCustomHardwareLinkAttachment: A
      GoogleCloudNetworkconnectivityV1betaCustomHardwareLinkAttachment
      resource to be passed as the request body.
    name: Identifier. The name of a CustomHardwareLinkAttachment. Format: `pro
      jects/{project}/locations/{location}/customHardwareLinkAttachments/{cust
      om_hardware_link_attachment}`.
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes since the first
      request. For example, consider a situation where you make an initial
      request and the request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the CustomHardwareLinkAttachment resource by the update.
      The fields specified in the update_mask are relative to the resource,
      not the full request. A field will be overwritten if it is in the mask.
      If the user does not provide a mask then all fields will be overwritten.
  """
    googleCloudNetworkconnectivityV1betaCustomHardwareLinkAttachment = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaCustomHardwareLinkAttachment', 1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    updateMask = _messages.StringField(4)
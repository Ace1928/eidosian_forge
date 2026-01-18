from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsPatchRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateCloudsPatchRequest object.

  Fields:
    name: Output only. The resource name of this private cloud. Resource names
      are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1-a/privateClouds/my-cloud`
    privateCloud: A PrivateCloud resource to be passed as the request body.
    requestId: Optional. The request ID must be a valid UUID with the
      exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the `PrivateCloud` resource by the update. The fields
      specified in `updateMask` are relative to the resource, not the full
      request. A field will be overwritten if it is in the mask. If the user
      does not provide a mask then all fields will be overwritten.
  """
    name = _messages.StringField(1, required=True)
    privateCloud = _messages.MessageField('PrivateCloud', 2)
    requestId = _messages.StringField(3)
    updateMask = _messages.StringField(4)
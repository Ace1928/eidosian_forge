from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsUpgradesPatchRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateCloudsUpgradesPatchRequest object.

  Fields:
    name: Output only. The resource name of the private cloud `Upgrade`.
      Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-west1-a/privateClouds/my-
      cloud/upgrades/my-upgrade`
    requestId: Optional. A request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server will know
      to ignore the request if it has already been completed. The server
      guarantees that a request doesn't result in creation of duplicate
      commitments for at least 60 minutes. For example, consider a situation
      where you make an initial request and the request times out. If you make
      the request again with the same request ID, the server can check if
      original operation with the same request ID was received, and if so,
      will ignore the second request. This prevents clients from accidentally
      creating duplicate commitments. The request ID must be a valid UUID with
      the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the `Upgrade` resource by the update. The fields
      specified in the `update_mask` are relative to the resource, not the
      full request. A field will be overwritten if it is in the mask. If the
      user does not provide a mask then all fields will be overwritten.
    upgrade: A Upgrade resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)
    updateMask = _messages.StringField(3)
    upgrade = _messages.MessageField('Upgrade', 4)
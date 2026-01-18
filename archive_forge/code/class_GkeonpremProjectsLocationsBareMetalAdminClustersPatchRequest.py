from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalAdminClustersPatchRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalAdminClustersPatchRequest object.

  Fields:
    bareMetalAdminCluster: A BareMetalAdminCluster resource to be passed as
      the request body.
    name: Immutable. The bare metal admin cluster resource name.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the BareMetalAdminCluster resource by the update. The
      fields specified in the update_mask are relative to the resource, not
      the full request. A field will be overwritten if it is in the mask. If
      the user does not provide a mask then all populated fields in the
      BareMetalAdminCluster message will be updated. Empty fields will be
      ignored unless a field mask is used.
    validateOnly: Validate the request without actually doing any updates.
  """
    bareMetalAdminCluster = _messages.MessageField('BareMetalAdminCluster', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsPatchRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandalo
  neNodePoolsPatchRequest object.

  Fields:
    allowMissing: If set to true, and the bare metal standalone node pool is
      not found, the request will create a new bare metal standalone node pool
      with the provided configuration. The user must have both create and
      update permission to call Update with allow_missing set to true.
    bareMetalStandaloneNodePool: A BareMetalStandaloneNodePool resource to be
      passed as the request body.
    name: Immutable. The bare metal standalone node pool resource name.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the BareMetalStandaloneNodePool resource by the update.
      The fields specified in the update_mask are relative to the resource,
      not the full request. A field will be overwritten if it is in the mask.
    validateOnly: Validate the request without actually doing any updates.
  """
    allowMissing = _messages.BooleanField(1)
    bareMetalStandaloneNodePool = _messages.MessageField('BareMetalStandaloneNodePool', 2)
    name = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)
    validateOnly = _messages.BooleanField(5)
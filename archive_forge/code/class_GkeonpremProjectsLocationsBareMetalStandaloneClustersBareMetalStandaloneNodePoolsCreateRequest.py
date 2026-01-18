from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsCreateRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandalo
  neNodePoolsCreateRequest object.

  Fields:
    bareMetalStandaloneNodePool: A BareMetalStandaloneNodePool resource to be
      passed as the request body.
    bareMetalStandaloneNodePoolId: The ID to use for the node pool, which will
      become the final component of the node pool's resource name. This value
      must be up to 63 characters, and valid characters are /a-z-/. The value
      must not be permitted to be a UUID (or UUID-like: anything matching
      /^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i).
    parent: Required. The parent resource where this node pool will be
      created. projects/{project}/locations/{location}/bareMetalStandaloneClus
      ters/{cluster}
    validateOnly: If set, only validate the request, but do not actually
      create the node pool.
  """
    bareMetalStandaloneNodePool = _messages.MessageField('BareMetalStandaloneNodePool', 1)
    bareMetalStandaloneNodePoolId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)
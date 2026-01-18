from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsCreateRequest(_messages.Message):
    """A GkeonpremProjectsLocationsVmwareClustersVmwareNodePoolsCreateRequest
  object.

  Fields:
    parent: Required. The parent resource where this node pool will be
      created.
      projects/{project}/locations/{location}/vmwareClusters/{cluster}
    validateOnly: If set, only validate the request, but do not actually
      create the node pool.
    vmwareNodePool: A VmwareNodePool resource to be passed as the request
      body.
    vmwareNodePoolId: The ID to use for the node pool, which will become the
      final component of the node pool's resource name. This value must be up
      to 40 characters and follow RFC-1123
      (https://tools.ietf.org/html/rfc1123) format. The value must not be
      permitted to be a UUID (or UUID-like: anything matching
      /^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i).
  """
    parent = _messages.StringField(1, required=True)
    validateOnly = _messages.BooleanField(2)
    vmwareNodePool = _messages.MessageField('VmwareNodePool', 3)
    vmwareNodePoolId = _messages.StringField(4)
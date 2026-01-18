from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsVmwareClustersCreateRequest(_messages.Message):
    """A GkeonpremProjectsLocationsVmwareClustersCreateRequest object.

  Fields:
    parent: Required. The parent of the project and location where this
      cluster is created in. Format: "projects/{project}/locations/{location}"
    validateOnly: Validate the request without actually doing any updates.
    vmwareCluster: A VmwareCluster resource to be passed as the request body.
    vmwareClusterId: User provided identifier that is used as part of the
      resource name; This value must be up to 40 characters and follow
      RFC-1123 (https://tools.ietf.org/html/rfc1123) format.
  """
    parent = _messages.StringField(1, required=True)
    validateOnly = _messages.BooleanField(2)
    vmwareCluster = _messages.MessageField('VmwareCluster', 3)
    vmwareClusterId = _messages.StringField(4)
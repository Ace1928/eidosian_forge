from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalAdminClustersCreateRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalAdminClustersCreateRequest object.

  Fields:
    bareMetalAdminCluster: A BareMetalAdminCluster resource to be passed as
      the request body.
    bareMetalAdminClusterId: Required. User provided identifier that is used
      as part of the resource name; must conform to RFC-1034 and additionally
      restrict to lower-cased letters. This comes out roughly to:
      /^a-z+[a-z0-9]$/
    parent: Required. The parent of the project and location where the cluster
      is created in. Format: "projects/{project}/locations/{location}"
    validateOnly: Validate the request without actually doing any updates.
  """
    bareMetalAdminCluster = _messages.MessageField('BareMetalAdminCluster', 1)
    bareMetalAdminClusterId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)
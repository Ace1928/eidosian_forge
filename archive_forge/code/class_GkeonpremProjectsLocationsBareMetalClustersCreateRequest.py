from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalClustersCreateRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalClustersCreateRequest object.

  Fields:
    bareMetalCluster: A BareMetalCluster resource to be passed as the request
      body.
    bareMetalClusterId: Required. User provided identifier that is used as
      part of the resource name; must conform to RFC-1034 and additionally
      restrict to lower-cased letters. This comes out roughly to:
      /^a-z+[a-z0-9]$/
    parent: Required. The parent of the project and location where the cluster
      is created in. Format: "projects/{project}/locations/{location}"
    validateOnly: Validate the request without actually doing any updates.
  """
    bareMetalCluster = _messages.MessageField('BareMetalCluster', 1)
    bareMetalClusterId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)
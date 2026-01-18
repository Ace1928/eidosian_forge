from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsBareMetalAdminClustersEnrollRequest(_messages.Message):
    """A GkeonpremProjectsLocationsBareMetalAdminClustersEnrollRequest object.

  Fields:
    enrollBareMetalAdminClusterRequest: A EnrollBareMetalAdminClusterRequest
      resource to be passed as the request body.
    parent: Required. The parent of the project and location where the cluster
      is enrolled in. Format: "projects/{project}/locations/{location}"
  """
    enrollBareMetalAdminClusterRequest = _messages.MessageField('EnrollBareMetalAdminClusterRequest', 1)
    parent = _messages.StringField(2, required=True)
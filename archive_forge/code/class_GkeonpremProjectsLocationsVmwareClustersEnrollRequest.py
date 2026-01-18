from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsVmwareClustersEnrollRequest(_messages.Message):
    """A GkeonpremProjectsLocationsVmwareClustersEnrollRequest object.

  Fields:
    enrollVmwareClusterRequest: A EnrollVmwareClusterRequest resource to be
      passed as the request body.
    parent: Required. The parent of the project and location where the cluster
      is Enrolled in. Format: "projects/{project}/locations/{location}"
  """
    enrollVmwareClusterRequest = _messages.MessageField('EnrollVmwareClusterRequest', 1)
    parent = _messages.StringField(2, required=True)
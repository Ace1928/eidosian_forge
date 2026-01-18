from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupsResetCloudAdminCredentialsRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupsResetCloudAdminCredentialsRequest
  object.

  Fields:
    clusterGroup: Required. The resource name of the cluster group on which to
      reset the **CloudAdmin** password. For example, projects/PROJECT-NUMBER
      /locations/us-central1/clusterGroups/MY_GROUP
    resetCloudAdminCredentialsRequest: A ResetCloudAdminCredentialsRequest
      resource to be passed as the request body.
  """
    clusterGroup = _messages.StringField(1, required=True)
    resetCloudAdminCredentialsRequest = _messages.MessageField('ResetCloudAdminCredentialsRequest', 2)
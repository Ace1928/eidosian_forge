from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def GetCreateClusterRequest(self, cluster, project_id, region, request_id, action_on_failed_primary_workers=None):
    """Gets the CreateClusterRequest for the appropriate api version.

    Args :
      cluster : Dataproc cluster to be created.
      project_id: The ID of the Google Cloud Platform project that the cluster
      belongs to.
      region : The Dataproc region in which to handle the request.
      request_id : A unique ID used to identify the request.
      action_on_failed_primary_workers : Supported only for v1 api.

    Raises :
      ValueError : if non-None action_on_failed_primary_workers is passed for
      v1beta2 api.

    Returns :
      DataprocProjectsRegionsClustersCreateRequest
    """
    if action_on_failed_primary_workers is None:
        return self.messages.DataprocProjectsRegionsClustersCreateRequest(cluster=cluster, projectId=project_id, region=region, requestId=request_id)
    if self.api_version == 'v1beta2':
        raise ValueError('action_on_failed_primary_workers is not supported for v1beta2 api')
    return self.messages.DataprocProjectsRegionsClustersCreateRequest(cluster=cluster, projectId=project_id, region=region, requestId=request_id, actionOnFailedPrimaryWorkers=action_on_failed_primary_workers)
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import client
from googlecloudsdk.api_lib.container.gkemulticloud import update_mask
from googlecloudsdk.command_lib.container.aws import flags as aws_flags
from googlecloudsdk.command_lib.container.gkemulticloud import flags
class NodePoolsClient(_AwsClientBase):
    """Client for AWS node pools in the gkemulticloud API."""

    def __init__(self, **kwargs):
        super(NodePoolsClient, self).__init__(**kwargs)
        self._service = self._client.projects_locations_awsClusters_awsNodePools
        self._list_result_field = 'awsNodePools'

    def Create(self, node_pool_ref, args):
        """Creates an node pool in an Anthos cluster on AWS."""
        req = self._messages.GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsCreateRequest(awsNodePoolId=node_pool_ref.awsNodePoolsId, googleCloudGkemulticloudV1AwsNodePool=self._NodePool(node_pool_ref, args), parent=node_pool_ref.Parent().RelativeName(), validateOnly=flags.GetValidateOnly(args))
        return self._service.Create(req)

    def Update(self, node_pool_ref, args):
        """Updates a node pool in an Anthos cluster on AWS."""
        req = self._messages.GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsPatchRequest(googleCloudGkemulticloudV1AwsNodePool=self._NodePool(node_pool_ref, args), name=node_pool_ref.RelativeName(), updateMask=update_mask.GetUpdateMask(args, update_mask.AWS_NODEPOOL_ARGS_TO_UPDATE_MASKS), validateOnly=flags.GetValidateOnly(args))
        return self._service.Patch(req)

    def Rollback(self, node_pool_ref, args):
        """Rolls back a node pool in an Anthos cluster on AWS."""
        req = self._messages.GkemulticloudProjectsLocationsAwsClustersAwsNodePoolsRollbackRequest(name=node_pool_ref.RelativeName(), googleCloudGkemulticloudV1RollbackAwsNodePoolUpdateRequest=self._messages.GoogleCloudGkemulticloudV1RollbackAwsNodePoolUpdateRequest(respectPdb=flags.GetRespectPodDisruptionBudget(args)))
        return self._service.Rollback(req)
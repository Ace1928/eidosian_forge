from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages as protorpc_message
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import standalone_clusters
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages_module
class StandaloneNodePoolsClient(_StandaloneNodePoolsClient):
    """Client for node pools in Anthos clusters on bare metal standalone API."""

    def __init__(self, **kwargs):
        super(StandaloneNodePoolsClient, self).__init__(**kwargs)
        self._service = self._client.projects_locations_bareMetalStandaloneClusters_bareMetalStandaloneNodePools

    def List(self, location_ref: protorpc_message.Message, limit=None, page_size=None) -> protorpc_message.Message:
        """Lists Node Pools in the Anthos clusters on bare metal standalone API."""
        list_req = messages_module.GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsListRequest(parent=location_ref.RelativeName())
        return list_pager.YieldFromList(self._service, list_req, field='bareMetalStandaloneNodePools', batch_size=page_size, limit=limit, batch_size_attribute='pageSize')

    def Describe(self, resource_ref):
        """Gets a GKE On-Prem Bare Metal API standalone node pool resource."""
        req = messages_module.GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsGetRequest(name=resource_ref.RelativeName())
        return self._service.Get(req)

    def Enroll(self, args: parser_extensions.Namespace) -> protorpc_message.Message:
        """Enrolls an Anthos On-Prem Bare Metal API standalone node pool resource.

    Args:
      args: parser_extensions.Namespace, known args specified on the command
        line.

    Returns:
      (Operation) The response message.
    """
        req = messages_module.GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsEnrollRequest(enrollBareMetalStandaloneNodePoolRequest=messages_module.EnrollBareMetalStandaloneNodePoolRequest(bareMetalStandaloneNodePoolId=self._standalone_node_pool_id(args), validateOnly=self.GetFlag(args, 'validate_only')), parent=self._standalone_node_pool_parent(args))
        return self._service.Enroll(req)

    def Unenroll(self, args: parser_extensions.Namespace) -> protorpc_message.Message:
        """Unenrolls an Anthos On-Prem bare metal API standalone node pool resource.

    Args:
      args: parser_extensions.Namespace, known args specified on the command
        line.

    Returns:
      (Operation) The response message.
    """
        kwargs = {'allowMissing': self.GetFlag(args, 'allow_missing'), 'name': self._standalone_node_pool_name(args), 'validateOnly': self.GetFlag(args, 'validate_only')}
        req = messages_module.GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsUnenrollRequest(**kwargs)
        return self._service.Unenroll(req)

    def Delete(self, args: parser_extensions.Namespace) -> protorpc_message.Message:
        """Deletes a GKE On-Prem Bare Metal API standalone node pool resource."""
        kwargs = {'name': self._standalone_node_pool_name(args), 'allowMissing': self.GetFlag(args, 'allow_missing'), 'validateOnly': self.GetFlag(args, 'validate_only'), 'ignoreErrors': self.GetFlag(args, 'ignore_errors')}
        req = messages_module.GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsDeleteRequest(**kwargs)
        return self._service.Delete(req)

    def Create(self, args: parser_extensions.Namespace):
        """Creates a GKE On-Prem Bare Metal API standalone node pool resource."""
        node_pool_ref = self._node_pool_ref(args)
        kwargs = {'parent': node_pool_ref.Parent().RelativeName(), 'validateOnly': self.GetFlag(args, 'validate_only'), 'bareMetalStandaloneNodePool': self._bare_metal_standalone_node_pool(args), 'bareMetalStandaloneNodePoolId': self._standalone_node_pool_id(args)}
        req = messages_module.GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsCreateRequest(**kwargs)
        return self._service.Create(req)

    def Update(self, args: parser_extensions.Namespace) -> protorpc_message.Message:
        """Updates a GKE On-Prem Bare Metal API standalone node pool resource."""
        req = messages_module.GkeonpremProjectsLocationsBareMetalStandaloneClustersBareMetalStandaloneNodePoolsPatchRequest(allowMissing=self.GetFlag(args, 'allow_missing'), name=self._standalone_node_pool_name(args), updateMask=update_mask.get_update_mask(args, update_mask.BARE_METAL_STANDALONE_NODE_POOL_ARGS_TO_UPDATE_MASKS), validateOnly=self.GetFlag(args, 'validate_only'), bareMetalStandaloneNodePool=self._bare_metal_standalone_node_pool(args))
        return self._service.Patch(req)
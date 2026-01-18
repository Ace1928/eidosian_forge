from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
class StandaloneClustersClient(_BareMetalStandaloneClusterClient):
    """Client for clusters in gkeonprem bare metal standalone API."""

    def __init__(self, **kwargs):
        super(StandaloneClustersClient, self).__init__(**kwargs)
        self._service = self._client.projects_locations_bareMetalStandaloneClusters

    def List(self, location_ref, limit=None, page_size=None):
        """Lists Clusters in the GKE On-Prem Bare Metal Standalone API."""
        list_req = messages.GkeonpremProjectsLocationsBareMetalStandaloneClustersListRequest(parent=location_ref.RelativeName())
        return list_pager.YieldFromList(self._service, list_req, field='bareMetalStandaloneClusters', batch_size=page_size, limit=limit, batch_size_attribute='pageSize')

    def Describe(self, resource_ref):
        """Gets a GKE On-Prem Bare Metal Standalone API cluster resource."""
        req = messages.GkeonpremProjectsLocationsBareMetalStandaloneClustersGetRequest(name=resource_ref.RelativeName())
        return self._service.Get(req)

    def Enroll(self, args: parser_extensions.Namespace):
        """Enrolls an existing bare metal standalone cluster to the GKE on-prem API within a given project and location."""
        kwargs = {'membership': self._standalone_cluster_membership_name(args), 'bareMetalStandaloneClusterId': self._standalone_cluster_id(args)}
        req = messages.GkeonpremProjectsLocationsBareMetalStandaloneClustersEnrollRequest(parent=self._standalone_cluster_parent(args), enrollBareMetalStandaloneClusterRequest=messages.EnrollBareMetalStandaloneClusterRequest(**kwargs))
        return self._service.Enroll(req)

    def Unenroll(self, args: parser_extensions.Namespace):
        """Unenrolls an Anthos on bare metal standalone cluster."""
        kwargs = {'name': self._standalone_cluster_name(args), 'allowMissing': getattr(args, 'allow_missing', None), 'ignoreErrors': getattr(args, 'ignore_errors', None)}
        req = messages.GkeonpremProjectsLocationsBareMetalStandaloneClustersUnenrollRequest(**kwargs)
        return self._service.Unenroll(req)

    def QueryVersionConfig(self, args: parser_extensions.Namespace):
        """Query Anthos on bare metal standalone cluster version configuration."""
        kwargs = {'upgradeConfig_clusterName': self._standalone_cluster_name(args), 'parent': self._location_ref(args).RelativeName()}
        encoding.AddCustomJsonFieldMapping(messages.GkeonpremProjectsLocationsBareMetalStandaloneClustersQueryVersionConfigRequest, 'upgradeConfig_clusterName', 'upgradeConfig.clusterName')
        req = messages.GkeonpremProjectsLocationsBareMetalStandaloneClustersQueryVersionConfigRequest(**kwargs)
        return self._service.QueryVersionConfig(req)

    def Update(self, args: parser_extensions.Namespace):
        """Updates an Anthos on bare metal standalone cluster."""
        kwargs = {'name': self._standalone_cluster_name(args), 'allowMissing': getattr(args, 'allow_missing', None), 'updateMask': update_mask.get_update_mask(args, update_mask.BARE_METAL_STANDALONE_CLUSTER_ARGS_TO_UPDATE_MASKS), 'validateOnly': getattr(args, 'validate_only', False), 'bareMetalStandaloneCluster': self._bare_metal_standalone_cluster(args)}
        req = messages.GkeonpremProjectsLocationsBareMetalStandaloneClustersPatchRequest(**kwargs)
        return self._service.Patch(req)
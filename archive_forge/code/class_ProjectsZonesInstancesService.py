from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1beta import osconfig_v1beta_messages as messages
class ProjectsZonesInstancesService(base_api.BaseApiService):
    """Service class for the projects_zones_instances resource."""
    _NAME = 'projects_zones_instances'

    def __init__(self, client):
        super(OsconfigV1beta.ProjectsZonesInstancesService, self).__init__(client)
        self._upload_configs = {}

    def LookupEffectiveGuestPolicy(self, request, global_params=None):
        """Lookup the effective guest policy that applies to a VM instance. This lookup merges all policies that are assigned to the instance ancestry.

      Args:
        request: (OsconfigProjectsZonesInstancesLookupEffectiveGuestPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EffectiveGuestPolicy) The response message.
      """
        config = self.GetMethodConfig('LookupEffectiveGuestPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    LookupEffectiveGuestPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/zones/{zonesId}/instances/{instancesId}:lookupEffectiveGuestPolicy', http_method='POST', method_id='osconfig.projects.zones.instances.lookupEffectiveGuestPolicy', ordered_params=['instance'], path_params=['instance'], query_params=[], relative_path='v1beta/{+instance}:lookupEffectiveGuestPolicy', request_field='lookupEffectiveGuestPolicyRequest', request_type_name='OsconfigProjectsZonesInstancesLookupEffectiveGuestPolicyRequest', response_type_name='EffectiveGuestPolicy', supports_download=False)
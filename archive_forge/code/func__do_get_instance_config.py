from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import path_simplifier
import six
def _do_get_instance_config(self, igm_ref, instance_ref):
    """Returns instance config for given instance."""
    instance_name = path_simplifier.Name(six.text_type(instance_ref))
    filter_param = 'name eq {0}'.format(instance_name)
    if igm_ref.Collection() == 'compute.instanceGroupManagers':
        service = self._client.apitools_client.instanceGroupManagers
        request = self._client.messages.ComputeInstanceGroupManagersListPerInstanceConfigsRequest(instanceGroupManager=igm_ref.Name(), project=igm_ref.project, zone=igm_ref.zone, filter=filter_param, maxResults=1)
    elif igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
        service = self._client.apitools_client.regionInstanceGroupManagers
        request = self._client.messages.ComputeRegionInstanceGroupManagersListPerInstanceConfigsRequest(instanceGroupManager=igm_ref.Name(), project=igm_ref.project, region=igm_ref.region, filter=filter_param, maxResults=1)
    else:
        raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))
    per_instance_configs = service.ListPerInstanceConfigs(request).items
    if per_instance_configs:
        return per_instance_configs[0]
    else:
        return None
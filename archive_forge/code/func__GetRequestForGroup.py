from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute.instance_groups.managed import wait_info
from googlecloudsdk.core import log
from googlecloudsdk.core.util import retry
def _GetRequestForGroup(client, group_ref):
    """Executes a request for a group - either zonal or regional one."""
    if group_ref.Collection() == 'compute.instanceGroupManagers':
        service = client.apitools_client.instanceGroupManagers
        request = service.GetRequestType('Get')(instanceGroupManager=group_ref.Name(), zone=group_ref.zone, project=group_ref.project)
    elif group_ref.Collection() == 'compute.regionInstanceGroupManagers':
        service = client.apitools_client.regionInstanceGroupManagers
        request = service.GetRequestType('Get')(instanceGroupManager=group_ref.Name(), region=group_ref.region, project=group_ref.project)
    else:
        raise ValueError('Unknown reference type {0}'.format(group_ref.Collection()))
    return (service, request)
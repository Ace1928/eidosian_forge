from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.core import properties
def _TransformState(resource):
    state = resource.get('state', '')
    if 'instanceDetailsSummary' in resource:
        num_instances_pending_reboot = int(resource['instanceDetailsSummary'].get('instancesSucceededRebootRequired', '0'))
        if state == 'SUCCEEDED' and num_instances_pending_reboot > 0:
            return 'SUCCEEDED_INSTANCES_PENDING_REBOOT'
    return state
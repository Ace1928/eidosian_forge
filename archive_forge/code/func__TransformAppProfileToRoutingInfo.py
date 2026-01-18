from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import app_profiles
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
def _TransformAppProfileToRoutingInfo(app_profile):
    """Extracts the routing info from the app profile."""
    if 'singleClusterRouting' in app_profile and 'clusterId' in app_profile['singleClusterRouting']:
        return app_profile['singleClusterRouting']['clusterId']
    elif 'multiClusterRoutingUseAny' in app_profile:
        if 'clusterIds' in app_profile['multiClusterRoutingUseAny']:
            return ','.join(app_profile['multiClusterRoutingUseAny']['clusterIds'])
        return 'MULTI_CLUSTER_USE_ANY'
    return ''
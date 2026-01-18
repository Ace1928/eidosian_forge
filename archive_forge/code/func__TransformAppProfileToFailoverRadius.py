from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import app_profiles
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
def _TransformAppProfileToFailoverRadius(app_profile):
    """Extracts the failover radius from the app profile."""
    if 'multiClusterRoutingUseAny' in app_profile:
        if 'failoverRadius' in app_profile['multiClusterRoutingUseAny']:
            return app_profile['multiClusterRoutingUseAny']['failoverRadius']
    return ''
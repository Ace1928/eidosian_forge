from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ManagedInstanceGroupsInstanceConfigs(base.Group):
    """Manage instance-specific settings in a managed instance group."""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import filter_rewrite
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.network_endpoint_groups import flags
from googlecloudsdk.core.resource import resource_projection_spec
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class BetaListNetworkEndpoints(ListNetworkEndpoints):
    """List network endpoints in a network endpoint group."""
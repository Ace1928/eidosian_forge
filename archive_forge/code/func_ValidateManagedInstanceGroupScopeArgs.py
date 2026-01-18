from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from googlecloudsdk.command_lib.util import completers
import six
def ValidateManagedInstanceGroupScopeArgs(args, resources):
    """Validate arguments specifying scope of the managed instance group."""
    ignored_required_params = {'project': 'fake'}
    if args.zones and args.zone:
        raise exceptions.ConflictingArgumentsException('--zone', '--zones')
    zone_names = []
    for zone in args.zones:
        zone_ref = resources.Parse(zone, collection='compute.zones', params=ignored_required_params)
        zone_names.append(zone_ref.Name())
    zone_regions = set([utils.ZoneNameToRegionName(z) for z in zone_names])
    if len(zone_regions) > 1:
        raise exceptions.InvalidArgumentException('--zones', 'All zones must be in the same region.')
    elif len(zone_regions) == 1 and args.region:
        zone_region = zone_regions.pop()
        region_ref = resources.Parse(args.region, collection='compute.regions', params=ignored_required_params)
        region = region_ref.Name()
        if zone_region != region:
            raise exceptions.InvalidArgumentException('--zones', 'Specified zones not in specified region.')
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import random
import re
import string
import sys
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.compute.managed_instance_groups import update_instances_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _ComputeInstanceGroupSize(items, client, resources):
    """Add information about Instance Group size."""
    errors = []
    zone_refs = [resources.Parse(mig['zone'], params={'project': properties.VALUES.core.project.GetOrFail}, collection='compute.zones') for mig in items if 'zone' in mig]
    region_refs = [resources.Parse(mig['region'], params={'project': properties.VALUES.core.project.GetOrFail}, collection='compute.regions') for mig in items if 'region' in mig]
    zonal_instance_groups = []
    for project, zone_refs in six.iteritems(GroupByProject(zone_refs)):
        zonal_instance_groups.extend(lister.GetZonalResources(service=client.apitools_client.instanceGroups, project=project, requested_zones=set([zone.zone for zone in zone_refs]), filter_expr=None, http=client.apitools_client.http, batch_url=client.batch_url, errors=errors))
    regional_instance_groups = []
    if getattr(client.apitools_client, 'regionInstanceGroups', None):
        for project, region_refs in six.iteritems(GroupByProject(region_refs)):
            regional_instance_groups.extend(lister.GetRegionalResources(service=client.apitools_client.regionInstanceGroups, project=project, requested_regions=set([region.region for region in region_refs]), filter_expr=None, http=client.apitools_client.http, batch_url=client.batch_url, errors=errors))
    instance_groups = zonal_instance_groups + regional_instance_groups
    instance_group_uri_to_size = {ig.selfLink: ig.size for ig in instance_groups}
    if errors:
        utils.RaiseToolException(errors)
    for item in items:
        self_link = item['selfLink']
        gm_self_link = self_link.replace('/instanceGroupManagers/', '/instanceGroups/')
        item['size'] = str(instance_group_uri_to_size.get(gm_self_link, ''))
        yield item
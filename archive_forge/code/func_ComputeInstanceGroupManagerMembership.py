from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
from six.moves import range  # pylint: disable=redefined-builtin
def ComputeInstanceGroupManagerMembership(compute_holder, items, filter_mode=InstanceGroupFilteringMode.ALL_GROUPS):
    """Add information if instance group is managed.

  Args:
    compute_holder: ComputeApiHolder, The compute API holder
    items: list of instance group messages,
    filter_mode: InstanceGroupFilteringMode, managed/unmanaged filtering options
  Returns:
    list of instance groups with computed dynamic properties
  """
    client = compute_holder.client
    resources = compute_holder.resources
    errors = []
    items = list(items)
    zone_links = set([ig['zone'] for ig in items if 'zone' in ig])
    project_to_zones = {}
    for zone in zone_links:
        zone_ref = resources.Parse(zone, params={'project': properties.VALUES.core.project.GetOrFail}, collection='compute.zones')
        if zone_ref.project not in project_to_zones:
            project_to_zones[zone_ref.project] = set()
        project_to_zones[zone_ref.project].add(zone_ref.zone)
    zonal_instance_group_managers = []
    for project, zones in six.iteritems(project_to_zones):
        zonal_instance_group_managers.extend(lister.GetZonalResources(service=client.apitools_client.instanceGroupManagers, project=project, requested_zones=zones, filter_expr=None, http=client.apitools_client.http, batch_url=client.batch_url, errors=errors))
    regional_instance_group_managers = []
    if hasattr(client.apitools_client, 'regionInstanceGroups'):
        region_links = set([ig['region'] for ig in items if 'region' in ig])
        project_to_regions = {}
        for region in region_links:
            region_ref = resources.Parse(region, collection='compute.regions')
            if region_ref.project not in project_to_regions:
                project_to_regions[region_ref.project] = set()
            project_to_regions[region_ref.project].add(region_ref.region)
        for project, regions in six.iteritems(project_to_regions):
            regional_instance_group_managers.extend(lister.GetRegionalResources(service=client.apitools_client.regionInstanceGroupManagers, project=project, requested_regions=regions, filter_expr=None, http=client.apitools_client.http, batch_url=client.batch_url, errors=errors))
    instance_group_managers = list(zonal_instance_group_managers) + list(regional_instance_group_managers)
    instance_group_managers_refs = set([path_simplifier.ScopedSuffix(igm.selfLink) for igm in instance_group_managers])
    if errors:
        utils.RaiseToolException(errors)
    results = []
    for item in items:
        self_link = item['selfLink']
        igm_self_link = self_link.replace('/instanceGroups/', '/instanceGroupManagers/')
        scoped_suffix = path_simplifier.ScopedSuffix(igm_self_link)
        is_managed = scoped_suffix in instance_group_managers_refs
        if is_managed and filter_mode == InstanceGroupFilteringMode.ONLY_UNMANAGED_GROUPS:
            continue
        elif not is_managed and filter_mode == InstanceGroupFilteringMode.ONLY_MANAGED_GROUPS:
            continue
        item['isManaged'] = 'Yes' if is_managed else 'No'
        if is_managed:
            item['instanceGroupManagerUri'] = igm_self_link
        results.append(item)
    return results
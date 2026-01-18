from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _ParseDisk(resources, disk, sample, project_name, project_to_region, region, region_name, replica_zones):
    """Parse single disk reference."""
    disk_resource = resources.Parse(disk, params={'region': region_name, 'project': project_name}, collection='compute.regionDisks')
    current_project = disk_resource.project
    if current_project not in project_to_region:
        project_to_region[current_project] = _DeduceRegionInProject(resources, current_project, disk_resource, sample, region, region_name, replica_zones)
    result_disk = resources.Parse(disk, collection='compute.regionDisks', params={'region': project_to_region[current_project], 'project': current_project})
    if result_disk.region != project_to_region[current_project]:
        raise exceptions.InvalidArgumentException('--replica-zones', 'Region from [DISK_NAME] ({}) is different from [--replica-zones] ({}).'.format(result_disk.SelfLink(), project_to_region[current_project]))
    return result_disk
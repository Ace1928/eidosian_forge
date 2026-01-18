from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import gce as c_gce
import six
from six.moves import zip  # pylint: disable=redefined-builtin
def CreateZonalReferences(self, resource_names, zone_arg, resource_type=None, flag_names=None, region_filter=None):
    """Returns a list of resolved zonal resource references."""
    if flag_names is None:
        flag_names = ['--zone']
    if zone_arg:
        zone_ref = self.resources.Parse(zone_arg, params={'project': properties.VALUES.core.project.GetOrFail}, collection='compute.zones')
        zone_name = zone_ref.Name()
    else:
        zone_name = None
    return self.CreateScopedReferences(resource_names, scope_name='zone', scope_arg=zone_name, scope_service=self.compute.zones, resource_type=resource_type, flag_names=flag_names, prefix_filter=region_filter)
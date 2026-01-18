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
def CreateRegionalReferences(self, resource_names, region_arg, flag_names=None, resource_type=None):
    """Returns a list of resolved regional resource references."""
    if flag_names is None:
        flag_names = ['--region']
    if region_arg:
        region_ref = self.resources.Parse(region_arg, params={'project': properties.VALUES.core.project.GetOrFail}, collection='compute.regions')
        region_name = region_ref.Name()
    else:
        region_name = None
    return self.CreateScopedReferences(resource_names, scope_name='region', scope_arg=region_name, scope_service=self.compute.regions, flag_names=flag_names, resource_type=resource_type)
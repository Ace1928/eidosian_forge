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
def CreateGlobalReferences(self, resource_names, resource_type=None):
    """Returns a list of resolved global resource references."""
    resource_refs = []
    for resource_name in resource_names:
        resource_refs.append(self.resources.Parse(resource_name, params={'project': properties.VALUES.core.project.GetOrFail}, collection=utils.GetApiCollection(resource_type or self.resource_type)))
    return resource_refs
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.iam import util as iam_api
from googlecloudsdk.api_lib.resource_manager import tags
from googlecloudsdk.api_lib.resource_manager.exceptions import ResourceManagerError
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.resource_manager import endpoint_utils as endpoints
from googlecloudsdk.core import exceptions as core_exceptions
def GetCanonicalResourceName(resource_name, location, release_track):
    """Returns the correct canonical name for the given resource.

  Args:
    resource_name: name of the resource
    location: location in which the resource lives
    release_track: release stage of current endpoint

  Returns:
    resource_name: either the original resource name, or correct canonical name

  Raises:
    InvalidArgumentException: if the location is not specified
  """
    service_account_resource_name_pattern = 'iam.*/projects/[^/]+/serviceAccounts/([^/]+)'
    service_account_search = re.search(service_account_resource_name_pattern, resource_name)
    if service_account_search:
        service_account_name = service_account_search.group(1)
        if re.search('.*@.*.gserviceaccount.com', service_account_name):
            resource_name = resource_name.replace('serviceAccounts/%s' % service_account_name, 'serviceAccounts/%s' % _GetServiceAccountUniqueId(service_account_name))
        return resource_name
    gce_compute_instance_name_pattern = 'compute.googleapis.com/projects/([^/]+)/.*instances/([^/]+)'
    gce_search = re.search(gce_compute_instance_name_pattern, resource_name)
    if gce_search:
        if not location:
            raise exceptions.InvalidArgumentException('--location', 'Please specify an appropriate cloud location with the --location flag.')
        project_identifier, instance_identifier = (gce_search.group(1), gce_search.group(2))
        if re.search('([a-z]([-a-z0-9]*[a-z0-9])?)', instance_identifier):
            resource_name = resource_name.replace('instances/%s' % instance_identifier, 'instances/%s' % _GetGceInstanceCanonicalName(project_identifier, instance_identifier, location, release_track))
    return resource_name
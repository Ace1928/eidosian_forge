from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def ListUsableSubnets(self, project_ref, network_project, filter_arg):
    """List usable subnets for a given project.

    Args:
      project_ref: project where clusters will be created.
      network_project: project ID where clusters will be created.
      filter_arg: value of filter flag.

    Returns:
      Response containing the list of subnetworks and a next page token.
    """
    filters = []
    if network_project is not None:
        filters.append('networkProjectId=' + network_project)
    if filter_arg is not None:
        filters.append(filter_arg)
    filters = ' AND '.join(filters)
    req = self.messages.ContainerProjectsAggregatedUsableSubnetworksListRequest(parent=project_ref.RelativeName(), pageSize=500, filter=filters)
    return self.client.projects_aggregated_usableSubnetworks.List(req)
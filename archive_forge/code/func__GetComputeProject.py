from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.api_lib.compute import base_classes as compute_base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions as command_exceptions
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def _GetComputeProject(release_track):
    holder = compute_base_classes.ComputeApiHolder(release_track)
    client = holder.client
    project_ref = projects_util.ParseProject(properties.VALUES.core.project.GetOrFail())
    return client.MakeRequests([(client.apitools_client.projects, 'Get', client.messages.ComputeProjectsGetRequest(project=project_ref.projectId))])[0]
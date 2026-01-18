from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@staticmethod
def WorkspaceResourceName(name, location='global', use_number=False):
    """Builds a full Workspace name, using the core project property."""
    return util.WorkspaceResourceName(HubCommand.Project(use_number), name, location=location)
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
import stat
import textwrap
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def GetRunningInstances(self, client):
    """Returns a generator of all running instances in the project."""
    errors = []
    instances = lister.GetZonalResources(service=client.apitools_client.instances, project=properties.VALUES.core.project.GetOrFail(), requested_zones=None, filter_expr='status eq RUNNING', http=client.apitools_client.http, batch_url=client.batch_url, errors=errors)
    if errors:
        utils.RaiseToolException(errors, error_message='Could not fetch all instances:')
    return instances
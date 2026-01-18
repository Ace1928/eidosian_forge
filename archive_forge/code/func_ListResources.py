from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import errno
import io
import os
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.command_lib.util.resource_map.declarative import resource_name_translator
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.util import files
import six
def ListResources(self, project=None, organization=None, folder=None):
    """List all exportable resources.

    If parent (e.g. project, organization or folder) is passed then only list
    the exportable resources for that parent.

    Args:
      project: string, project to list exportable resources for.
      organization: string, organization to list exportable resources for.
      folder: string, folder to list exportable resources for.

    Returns:
      supported resources formatted output listing exportable resources.

    """
    if not (project or organization or folder):
        yaml_obj_list = yaml.load(self._CallPrintResources(output_format='yaml'), round_trip=True)
        return yaml_obj_list
    if project:
        msg_sfx = ' for project [{}]'.format(project)
    elif organization:
        msg_sfx = ' for organization [{}]'.format(organization)
    else:
        msg_sfx = ' for folder [{}]'.format(folder)
    with progress_tracker.ProgressTracker(message='Listing exportable resource types' + msg_sfx, aborted_message='Aborted Export.'):
        supported_kinds = self.ListSupportedResourcesForParent(project=project, organization=organization, folder=folder)
        supported_kinds = [x.AsDict() for x in supported_kinds]
        return supported_kinds
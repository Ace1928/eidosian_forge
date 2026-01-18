from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import properties
def GetSavedQueriesParent(organization, project, folder):
    """Get the parent name from organization Number, project Id, or folder Number."""
    if organization:
        return 'organizations/{0}'.format(organization)
    if folder:
        return 'folders/{0}'.format(folder)
    return 'projects/{0}'.format(project_util.GetProjectNumber(project))
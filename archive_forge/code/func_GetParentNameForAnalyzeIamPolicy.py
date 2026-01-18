from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import properties
def GetParentNameForAnalyzeIamPolicy(organization, project, folder, attribute='policy analysis scope'):
    """Gets the parent name from organization Id, project Id, or folder Id."""
    return GetParentNameForExport(organization, project, folder, attribute)
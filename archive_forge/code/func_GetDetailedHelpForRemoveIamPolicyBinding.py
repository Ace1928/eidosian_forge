from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import re
from apitools.base.py.exceptions import HttpForbiddenError
from googlecloudsdk.api_lib.cloudresourcemanager import organizations
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util
from googlecloudsdk.api_lib.iam import policies
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.projects import exceptions
from googlecloudsdk.core import resources
import six
def GetDetailedHelpForRemoveIamPolicyBinding():
    """Returns detailed_help for a remove-iam-policy-binding command."""
    detailed_help = iam_util.GetDetailedHelpForRemoveIamPolicyBinding('project', 'example-project-id-1', condition=True)
    detailed_help['DESCRIPTION'] += ' One binding consists of a member, a role and an optional condition.'
    detailed_help['API REFERENCE'] = 'This command uses the cloudresourcemanager/v1 API. The full documentation for this API can be found at: https://cloud.google.com/resource-manager'
    return detailed_help
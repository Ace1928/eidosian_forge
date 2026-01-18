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
def IsValidId(i):
    return 6 <= len(i) <= 30
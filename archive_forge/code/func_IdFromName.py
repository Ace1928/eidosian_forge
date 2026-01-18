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
def IdFromName(project_name):
    """Returns a candidate id for a new project with the given name.

  Args:
    project_name: Human-readable name of the project.

  Returns:
    A candidate project id, or 'None' if no reasonable candidate is found.
  """

    def SimplifyName(name):
        name = name.lower()
        name = re.sub('[^a-z0-9\\s/._-]', '', name, flags=re.U)
        name = re.sub('[\\s/._-]+', '-', name, flags=re.U)
        name = name.lstrip('-0123456789').rstrip('-')
        return name

    def CloudConsoleNowString():
        now = datetime.datetime.utcnow()
        return '{}{:02}'.format((now - _CLOUD_CONSOLE_LAUNCH_DATE).days, now.hour)

    def GenIds(name):
        base = SimplifyName(name)
        yield (base + '-' + CloudConsoleNowString())
        yield base

    def IsValidId(i):
        return 6 <= len(i) <= 30
    for i in GenIds(project_name):
        if IsValidId(i):
            return i
    return None
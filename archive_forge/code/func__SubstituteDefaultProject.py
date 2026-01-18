from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import log
import six
def _SubstituteDefaultProject(secret_version_ref, default_project_id, default_project_number):
    """Replaces the default project number in place of * or project ID.

  Args:
    secret_version_ref: Secret value reference.
    default_project_id: The project ID of the project to which the function is
      deployed.
    default_project_number: The project number of the project to which the
      function is deployed.

  Returns:
    Secret value reference with * or project ID replaced by the default project.
  """
    return re.sub('projects/([*]|{project_id})/'.format(project_id=default_project_id), 'projects/{project_number}/'.format(project_number=default_project_number), secret_version_ref)
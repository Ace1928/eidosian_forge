from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.util import retry
def GetValidatedProject(project_id):
    """Validate the project ID, if supplied, otherwise return the default project.

  Args:
    project_id: The ID of the project to validate. If None, gcloud's default
                project's ID will be returned.

  Returns:
    The validated project ID.
  """
    if project_id:
        properties.VALUES.core.project.Validate(project_id)
    else:
        project_id = properties.VALUES.core.project.Get(required=True)
    return project_id
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util
from googlecloudsdk.api_lib.resource_manager import operations
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.command_lib.projects import util as  projects_command_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
def PickProject(preselected=None):
    """Allows user to select a project.

  Args:
    preselected: str, use this value if not None

  Returns:
    str, project_id or None if was not selected.
  """
    project_ids = _GetProjectIds(limit=_PROJECT_LIST_LIMIT + 1)
    limit_exceeded = False
    if project_ids is not None and len(project_ids) > _PROJECT_LIST_LIMIT:
        limit_exceeded = True
    selected = None
    if preselected:
        project_id = preselected
    else:
        project_id = _PromptForProjectId(project_ids, limit_exceeded)
        if project_id is not _CREATE_PROJECT_SENTINEL:
            selected = project_id
    if not limit_exceeded:
        if project_ids is None or project_id in project_ids or project_id is None or selected:
            return project_id
    elif preselected and _IsExistingProject(preselected) or project_id is not _CREATE_PROJECT_SENTINEL:
        return project_id
    if project_id is _CREATE_PROJECT_SENTINEL:
        project_id = console_io.PromptResponse(_ENTER_PROJECT_ID_MESSAGE)
        if not project_id:
            return None
    else:
        if project_ids:
            message = '[{0}] is not one of your projects [{1}]. '.format(project_id, ','.join(project_ids))
        else:
            message = 'This account has no projects.'
        if not console_io.PromptContinue(message=message, prompt_string='Would you like to create it?'):
            return None
    return _CreateProject(project_id, project_ids)
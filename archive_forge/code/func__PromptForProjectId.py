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
def _PromptForProjectId(project_ids, limit_exceeded):
    """Prompt the user for a project ID, based on the list of available IDs.

  Also allows an option to create a project.

  Args:
    project_ids: list of str or None, the project IDs to prompt for. If this
      value is None, the listing was unsuccessful and we prompt the user
      free-form (and do not validate the input). If it's empty, we offer to
      create a project for the user.
    limit_exceeded: bool, whether or not the project list limit was reached. If
      this limit is reached, then user will be prompted with a choice to
      manually enter a project id, create a new project, or list all projects.

  Returns:
    str, the project ID to use, or _CREATE_PROJECT_SENTINEL (if a project should
      be created), or None
  """
    if project_ids is None:
        return console_io.PromptResponse('Enter project ID you would like to use:  ') or None
    elif not project_ids:
        if not console_io.PromptContinue('This account has no projects.', prompt_string='Would you like to create one?'):
            return None
        return _CREATE_PROJECT_SENTINEL
    elif limit_exceeded:
        idx = console_io.PromptChoice(['Enter a project ID', 'Create a new project', 'List projects'], message='This account has a lot of projects! Listing them all can take a while.')
        if idx is None:
            return None
        elif idx == 0:
            return console_io.PromptWithValidator(_IsExistingProject, 'Project ID does not exist or is not active.', 'Enter project ID you would like to use:  ', allow_invalid=True)
        elif idx == 1:
            return _CREATE_PROJECT_SENTINEL
        else:
            project_ids = _GetProjectIds()
    idx = console_io.PromptChoice(project_ids + ['Enter a project ID', 'Create a new project'], message='Pick cloud project to use: ', allow_freeform=True, freeform_suggester=usage_text.TextChoiceSuggester())
    if idx is None:
        return None
    elif idx == len(project_ids):
        return console_io.PromptWithValidator(_IsExistingProject, 'Project ID does not exist or is not active.', 'Enter project ID you would like to use:  ', allow_invalid=True)
    elif idx == len(project_ids) + 1:
        return _CREATE_PROJECT_SENTINEL
    return project_ids[idx]
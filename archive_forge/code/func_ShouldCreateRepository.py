from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py import exceptions as base_exceptions
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.artifacts import requests
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def ShouldCreateRepository(repo, skip_activation_prompt=False):
    """Checks for the existence of the provided repository.

  If the provided repository does not exist, the user will be prompted
  as to whether they would like to continue.

  Args:
    repo: googlecloudsdk.command_lib.artifacts.docker_util.DockerRepo defining
      the repository.
    skip_activation_prompt: bool determining if the client should prompt if the
      API isn't activated.

  Returns:
    A boolean indicating whether a repository needs to be created.
  """
    try:
        requests.GetRepository(repo.GetRepositoryName(), skip_activation_prompt)
        return False
    except base_exceptions.HttpForbiddenError:
        log.error('Permission denied while accessing Artifact Registry. Artifact Registry access is required to deploy from source.')
        raise
    except base_exceptions.HttpBadRequestError:
        log.error('Error in retrieving repository from Artifact Registry.')
        raise
    except base_exceptions.HttpNotFoundError:
        message = 'Deploying from source requires an Artifact Registry Docker repository to store built containers. A repository named [{name}] in region [{location}] will be created.'.format(name=repo.repo, location=repo.location)
        console_io.PromptContinue(message, cancel_on_no=True)
    return True
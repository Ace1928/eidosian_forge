from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import retry
def _BindRolesToServiceAccount(self, sa_email, roles):
    """Binds roles to the provided service account.

    Args:
      sa_email: str, the service account to bind roles to.
      roles: iterable, the roles to be bound to the service account.
    """
    formatted_roles = '\n'.join(['- {}'.format(role) for role in sorted(roles)])
    log.status.Print('To use Eventarc with Cloud Run for Anthos/GKE destinations, Eventarc Service Agent [{}] needs to be bound to the following required roles:\n{}'.format(sa_email, formatted_roles))
    console_io.PromptContinue(default=False, throw_if_unattended=True, prompt_string='\nWould you like to bind these roles?', cancel_on_no=True)
    project_ref = projects_util.ParseProject(properties.VALUES.core.project.Get(required=True))
    member_str = 'serviceAccount:{}'.format(sa_email)
    member_roles = [(member_str, role) for role in roles]
    self._AddIamPolicyBindingsWithRetry(project_ref, member_roles)
    log.status.Print('Roles successfully bound.')
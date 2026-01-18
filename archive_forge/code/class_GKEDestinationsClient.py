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
class GKEDestinationsClient(object):
    """Wrapper client for setting up Eventarc Cloud Run for Anthos/GKE destinations."""

    def __init__(self, release_track):
        self._api_version = common.GetApiVersion(release_track)
        client = apis.GetClientInstance(common.API_NAME, self._api_version)
        self._messages = client.MESSAGES_MODULE
        self._service = client.projects_locations_triggers

    def InitServiceAccount(self):
        """Force create the Eventarc P4SA, and grant IAM roles to it.

    1) First, trigger the P4SA JIT provision by trying to create an empty
    trigger, ignore the HttpBadRequestError exception, then call
    GenerateServiceIdentity to verify that P4SA creation is completed.
    2) Then grant necessary roles needed to the P4SA for creating GKE triggers.

    Raises:
      GKEDestinationInitializationError: P4SA failed to be created.
    """
        try:
            self._CreateEmptyTrigger()
        except apitools_exceptions.HttpBadRequestError:
            pass
        service_name = common.GetApiServiceName(self._api_version)
        p4sa_email = _GetOrCreateP4SA(service_name)
        if not p4sa_email:
            raise GKEDestinationInitializationError('Failed to initialize project for Cloud Run for Anthos/GKE destinations.')
        self._BindRolesToServiceAccount(p4sa_email, _ROLES)

    def _CreateEmptyTrigger(self):
        """Attempt to create an empty trigger in us-central1 to kick off P4SA JIT provision.

    The create request will always fail due to the empty trigger message
    payload, but it will trigger the P4SA JIT provision.

    Returns:
      A long-running operation for create.
    """
        project = properties.VALUES.core.project.Get(required=True)
        parent = 'projects/{}/locations/{}'.format(project, _LOCATION)
        req = self._messages.EventarcProjectsLocationsTriggersCreateRequest(parent=parent, triggerId=_TRIGGER_ID)
        return self._service.Create(req)

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

    @retry.RetryOnException(max_retrials=10, max_wait_ms=_MAX_WAIT_TIME_IN_MS, exponential_sleep_multiplier=1.6, sleep_ms=100, should_retry_if=_ShouldRetryHttpError)
    def _AddIamPolicyBindingsWithRetry(self, project_ref, member_roles):
        """Adds iam bindings to project_ref's iam policy, with retry.

    Args:
      project_ref: The project for the binding
      member_roles: List of 2-tuples of the form [(member, role), ...].

    Returns:
      The updated IAM Policy
    """
        return projects_api.AddIamPolicyBindings(project_ref, member_roles)
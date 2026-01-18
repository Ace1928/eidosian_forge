import typing
from typing import Mapping, Tuple
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import yaml
def ListUserWorkloadsSecrets(environment_ref: 'Resource', release_track: base.ReleaseTrack=base.ReleaseTrack.ALPHA) -> typing.Union[typing.List['composer_v1alpha2_messages.UserWorkloadsSecret'], typing.List['composer_v1beta1_messages.UserWorkloadsSecret'], typing.List['composer_v1_messages.UserWorkloadsSecret']]:
    """Calls the Composer Environments.ListUserWorkloadsSecrets method.

  Args:
    environment_ref: Resource, the Composer environment resource to list user
      workloads Secrets for.
    release_track: base.ReleaseTrack, the release track of the command. Will
      dictate which Composer client library will be used.

  Returns:
    list of user workloads Secrets.
  """
    message_module = api_util.GetMessagesModule(release_track=release_track)
    page_token = ''
    user_workloads_secrets = []
    while True:
        request_message = message_module.ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsListRequest(pageToken=page_token, parent=environment_ref.RelativeName())
        response = GetService(release_track=release_track).List(request_message)
        user_workloads_secrets.extend(response.userWorkloadsSecrets)
        if not response.nextPageToken:
            break
        page_token = response.nextPageToken
    return user_workloads_secrets
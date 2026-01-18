import typing
from typing import Mapping, Tuple
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import yaml
def CreateUserWorkloadsSecret(environment_ref: 'Resource', secret_file_path: str, release_track: base.ReleaseTrack=base.ReleaseTrack.ALPHA) -> typing.Union['composer_v1alpha2_messages.UserWorkloadsSecret', 'composer_v1beta1_messages.UserWorkloadsSecret', 'composer_v1_messages.UserWorkloadsSecret']:
    """Calls the Composer Environments.CreateUserWorkloadsSecret method.

  Args:
    environment_ref: Resource, the Composer environment resource to create a
      user workloads Secret for.
    secret_file_path: string, path to a local file with a Kubernetes Secret in
      yaml format.
    release_track: base.ReleaseTrack, the release track of the command. Will
      dictate which Composer client library will be used.

  Returns:
    UserWorkloadsSecret: the created user workloads Secret.

  Raises:
    command_util.InvalidUserInputError: if metadata.name was absent from the
    file.
  """
    message_module = api_util.GetMessagesModule(release_track=release_track)
    secret_name, secret_data = _ReadSecretFromFile(secret_file_path)
    user_workloads_secret_name = f'{environment_ref.RelativeName()}/userWorkloadsSecrets/{secret_name}'
    user_workloads_secret_data = api_util.DictToMessage(secret_data, message_module.UserWorkloadsSecret.DataValue)
    request_message = message_module.ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsCreateRequest(parent=environment_ref.RelativeName(), userWorkloadsSecret=message_module.UserWorkloadsSecret(name=user_workloads_secret_name, data=user_workloads_secret_data))
    return GetService(release_track=release_track).Create(request_message)
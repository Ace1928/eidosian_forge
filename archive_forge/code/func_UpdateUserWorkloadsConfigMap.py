import typing
from typing import Mapping, Tuple
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import yaml
def UpdateUserWorkloadsConfigMap(environment_ref: 'Resource', config_map_file_path: str, release_track: base.ReleaseTrack=base.ReleaseTrack.ALPHA) -> typing.Union['composer_v1alpha2_messages.UserWorkloadsConfigMap', 'composer_v1beta1_messages.UserWorkloadsConfigMap', 'composer_v1_messages.UserWorkloadsConfigMap']:
    """Calls the Composer Environments.UpdateUserWorkloadsConfigMap method.

  Args:
    environment_ref: Resource, the Composer environment resource to update a
      user workloads ConfigMap for.
    config_map_file_path: string, path to a local file with a Kubernetes
      ConfigMap in yaml format.
    release_track: base.ReleaseTrack, the release track of the command. Will
      dictate which Composer client library will be used.

  Returns:
    UserWorkloadsConfigMap: the updated user workloads ConfigMap.

  Raises:
    command_util.InvalidUserInputError: if metadata.name was absent from the
    file.
  """
    message_module = api_util.GetMessagesModule(release_track=release_track)
    config_map_name, config_map_data = _ReadConfigMapFromFile(config_map_file_path)
    user_workloads_config_map_name = f'{environment_ref.RelativeName()}/userWorkloadsConfigMaps/{config_map_name}'
    user_workloads_config_map_data = api_util.DictToMessage(config_map_data, message_module.UserWorkloadsConfigMap.DataValue)
    request_message = message_module.UserWorkloadsConfigMap(name=user_workloads_config_map_name, data=user_workloads_config_map_data)
    return GetService(release_track=release_track).Update(request_message)
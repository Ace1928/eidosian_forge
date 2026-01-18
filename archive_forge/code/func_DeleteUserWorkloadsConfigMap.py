import typing
from typing import Mapping, Tuple
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import yaml
def DeleteUserWorkloadsConfigMap(environment_ref: 'Resource', config_map_name: str, release_track: base.ReleaseTrack=base.ReleaseTrack.ALPHA):
    """Calls the Composer Environments.DeleteUserWorkloadsConfigMap method.

  Args:
    environment_ref: Resource, the Composer environment resource to delete a
      user workloads ConfigMap for.
    config_map_name: string, name of the Kubernetes ConfigMap.
    release_track: base.ReleaseTrack, the release track of the command. Will
      dictate which Composer client library will be used.
  """
    message_module = api_util.GetMessagesModule(release_track=release_track)
    user_workloads_config_map_name = f'{environment_ref.RelativeName()}/userWorkloadsConfigMaps/{config_map_name}'
    request_message = message_module.ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsDeleteRequest(name=user_workloads_config_map_name)
    GetService(release_track=release_track).Delete(request_message)
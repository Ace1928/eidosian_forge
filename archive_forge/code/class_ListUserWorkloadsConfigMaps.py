import textwrap
import typing
from typing import Sequence, Union
import frozendict
from googlecloudsdk.api_lib.composer import environments_user_workloads_config_maps_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import resource_args
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class ListUserWorkloadsConfigMaps(base.Command):
    """List user workloads ConfigMaps."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        resource_args.AddEnvironmentResourceArg(parser, 'to list user workloads ConfigMaps', positional=False)
        parser.display_info.AddFormat('table[box](name.segment(7),data)')

    def Run(self, args) -> Union[Sequence['composer_v1alpha2_messages.UserWorkloadsConfigMap'], Sequence['composer_v1beta1_messages.UserWorkloadsConfigMap'], Sequence['composer_v1_messages.UserWorkloadsConfigMap']]:
        env_resource = args.CONCEPTS.environment.Parse()
        return environments_user_workloads_config_maps_util.ListUserWorkloadsConfigMaps(env_resource, release_track=self.ReleaseTrack())
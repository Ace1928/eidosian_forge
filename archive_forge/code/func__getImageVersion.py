from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import environment_patch_util as patch_util
from googlecloudsdk.command_lib.composer import flags
from googlecloudsdk.command_lib.composer import image_versions_util as image_versions_command_util
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
def _getImageVersion(self, args, env_ref, env_obj):
    if (args.airflow_version or args.image_version) and image_versions_command_util.IsDefaultImageVersion(args.image_version):
        message = image_versions_command_util.BuildDefaultComposerVersionWarning(args.image_version, args.airflow_version)
        log.warning(message)
    if args.airflow_version:
        args.image_version = image_versions_command_util.ImageVersionFromAirflowVersion(args.airflow_version, env_obj.config.softwareConfig.imageVersion)
    if args.image_version:
        upgrade_validation = image_versions_command_util.IsValidImageVersionUpgrade(env_obj.config.softwareConfig.imageVersion, args.image_version)
        if not upgrade_validation.upgrade_valid:
            raise command_util.InvalidUserInputError(upgrade_validation.error)
    return args.image_version
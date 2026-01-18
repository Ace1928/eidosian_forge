from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import flags
from googlecloudsdk.command_lib.composer import image_versions_util
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
import six
def ParsePrivateEnvironmentConfigOptions(self, args, image_version):
    """Parses the options for Private Environment configuration."""
    if self.isComposer3(args):
        return
    if args.enable_private_environment and (not args.enable_ip_alias) and image_versions_util.IsImageVersionStringComposerV1(image_version):
        raise command_util.InvalidUserInputError(flags.PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-ip-alias', opt='enable-private-environment'))
    if args.enable_private_endpoint and (not args.enable_private_environment):
        raise command_util.InvalidUserInputError(flags.PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-private-environment', opt='enable-private-endpoint'))
    if args.enable_privately_used_public_ips and (not args.enable_private_environment):
        raise command_util.InvalidUserInputError(flags.PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-private-environment', opt='enable-privately-used-public-ips'))
    if args.master_ipv4_cidr and (not args.enable_private_environment):
        raise command_util.InvalidUserInputError(flags.PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-private-environment', opt='master-ipv4-cidr'))
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
def ValidateTriggererFlags(self, args):
    if args.image_version:
        triggerer_supported = image_versions_util.IsVersionTriggererCompatible(args.image_version)
        possible_args = {'enable-triggerer': args.enable_triggerer, 'triggerer-cpu': args.triggerer_cpu, 'triggerer-memory': args.triggerer_memory, 'triggerer-count': args.triggerer_count}
        for k, v in possible_args.items():
            if v and (not triggerer_supported):
                raise command_util.InvalidUserInputError(flags.INVALID_OPTION_FOR_MIN_IMAGE_VERSION_ERROR_MSG.format(opt=k, composer_version=flags.MIN_TRIGGERER_COMPOSER_VERSION, airflow_version=flags.MIN_TRIGGERER_AIRFLOW_VERSION))
    if not (args.enable_triggerer or (args.triggerer_count and args.triggerer_count > 0)):
        if args.triggerer_cpu:
            raise command_util.InvalidUserInputError(flags.ENABLED_TRIGGERER_IS_REQUIRED_MSG.format(opt='triggerer-cpu'))
        if args.triggerer_memory:
            raise command_util.InvalidUserInputError(flags.ENABLED_TRIGGERER_IS_REQUIRED_MSG.format(opt='triggerer-memory'))
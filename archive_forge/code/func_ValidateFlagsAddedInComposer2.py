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
def ValidateFlagsAddedInComposer2(self, args, is_composer_v1, release_track):
    """Raises InputError if flags from Composer v2 are used when creating v1."""
    if args.environment_size and is_composer_v1:
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V1_ERROR_MSG.format(opt='environment-size'))
    composer_v2_flag_used = args.scheduler_cpu or args.worker_cpu or args.web_server_cpu or args.scheduler_memory or args.worker_memory or args.web_server_memory or args.scheduler_storage or args.worker_storage or args.web_server_storage or args.min_workers or args.max_workers
    if composer_v2_flag_used and is_composer_v1:
        raise command_util.InvalidUserInputError('Workloads Config flags introduced in Composer 2.X cannot be used when creating Composer 1.X environments.')
    if args.enable_high_resilience and is_composer_v1:
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V1_ERROR_MSG.format(opt='enable-high-resilience'))
    if args.enable_logs_in_cloud_logging_only and is_composer_v1:
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V1_ERROR_MSG.format(opt='enable-logs-in-cloud-logging-only'))
    if args.disable_logs_in_cloud_logging_only and is_composer_v1:
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V1_ERROR_MSG.format(opt='disable-logs-in-cloud-logging-only'))
    if args.enable_cloud_data_lineage_integration and is_composer_v1:
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V1_ERROR_MSG.format(opt='enable-cloud-data-lineage-integration'))
    if args.disable_cloud_data_lineage_integration and is_composer_v1:
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V1_ERROR_MSG.format(opt='disable-cloud-data-lineage-integration'))
    if args.cloud_sql_preferred_zone and is_composer_v1:
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V1_ERROR_MSG.format(opt='cloud-sql-preferred-zone'))
    if args.connection_subnetwork and is_composer_v1:
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V1_ERROR_MSG.format(opt='connection-subnetwork'))
    if args.connection_subnetwork and (not args.enable_private_environment):
        raise command_util.InvalidUserInputError(flags.PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-private-environment', opt='connection-subnetwork'))
    if args.connection_type and is_composer_v1:
        raise command_util.InvalidUserInputError(_INVALID_OPTION_FOR_V1_ERROR_MSG.format(opt='connection-type'))
    if args.connection_type and (not args.enable_private_environment):
        raise command_util.InvalidUserInputError(flags.PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-private-environment', opt='connection-type'))
    if args.connection_type and args.connection_type == 'vpc-peering' and args.connection_subnetwork:
        raise command_util.InvalidUserInputError("You cannot specify a connection subnetwork if connection type 'VPC_PEERING' is selected.")
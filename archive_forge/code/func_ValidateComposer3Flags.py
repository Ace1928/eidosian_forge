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
def ValidateComposer3Flags(self, args):
    is_composer3 = self.isComposer3(args)
    forbidden_args = {'cloud-sql-ipv4-cidr': args.cloud_sql_ipv4_cidr, 'composer-network-ipv4-cidr': args.composer_network_ipv4_cidr, 'connection-subnetwork': args.connection_subnetwork, 'enable-private-endpoint': args.enable_private_endpoint, 'master-ipv4-cidr': args.master_ipv4_cidr}
    possible_args = {'support-web-server-plugins': args.support_web_server_plugins, 'dag-processor-cpu': args.dag_processor_cpu, 'dag-processor-memory': args.dag_processor_memory, 'dag-processor-count': args.dag_processor_count, 'dag-processor-storage': args.dag_processor_storage, 'network-attachment': args.network_attachment, 'composer-internal-ipv4-cidr-block': args.composer_internal_ipv4_cidr_block, 'enable-private-builds-only': args.enable_private_builds_only, 'disable-private-builds-only': args.disable_private_builds_only}
    for k, v in possible_args.items():
        if v is not None and (not is_composer3):
            raise command_util.InvalidUserInputError(flags.COMPOSER3_IS_REQUIRED_MSG.format(opt=k, composer_version=flags.MIN_COMPOSER3_VERSION))
    for k, v in forbidden_args.items():
        if v is not None and is_composer3:
            raise command_util.InvalidUserInputError(flags.COMPOSER3_IS_NOT_SUPPORTED_MSG.format(opt=k, composer_version=flags.MIN_COMPOSER3_VERSION))
    if args.network_attachment and (args.network or args.subnetwork):
        raise command_util.InvalidUserInputError('argument --network-attachment: At most one of --network-attachment | [--network : --subnetwork] can be specified')
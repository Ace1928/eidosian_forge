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
def AddComposer3Flags(parser):
    """Adds Composer 3 flags to the parser."""
    flags.SUPPORT_WEB_SERVER_PLUGINS.AddToParser(parser)
    dag_processor_params_group = parser.add_argument_group(flags.DAG_PROCESSOR_PARAMETERS_FLAG_GROUP_DESCRIPTION)
    flags.DAG_PROCESSOR_CPU.AddToParser(dag_processor_params_group)
    flags.DAG_PROCESSOR_COUNT.AddToParser(dag_processor_params_group)
    flags.DAG_PROCESSOR_MEMORY.AddToParser(dag_processor_params_group)
    flags.DAG_PROCESSOR_STORAGE.AddToParser(dag_processor_params_group)
    flags.COMPOSER_INTERNAL_IPV4_CIDR_FLAG.AddToParser(parser)
    private_builds_only_group = parser.add_mutually_exclusive_group()
    flags.ENABLE_PRIVATE_BUILDS_ONLY.AddToParser(private_builds_only_group)
    flags.DISABLE_PRIVATE_BUILDS_ONLY.AddToParser(private_builds_only_group)
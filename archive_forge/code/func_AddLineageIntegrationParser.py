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
def AddLineageIntegrationParser(parser):
    """Adds lineage integration flags to the parser."""
    lineage_params_group = parser.add_mutually_exclusive_group()
    flags.ENABLE_CLOUD_DATA_LINEAGE_INTEGRATION_FLAG.AddToParser(lineage_params_group)
    flags.DISABLE_CLOUD_DATA_LINEAGE_INTEGRATION_FLAG.AddToParser(lineage_params_group)
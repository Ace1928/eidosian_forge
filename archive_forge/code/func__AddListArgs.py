from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import operations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import endpoint_util
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import operations_util
def _AddListArgs(parser):
    list_format = '    table(\n        name.basename(),\n        metadata.operationType,\n        done\n    )\n'
    parser.display_info.AddFormat(list_format)
    flags.GetRegionArg().AddToParser(parser)
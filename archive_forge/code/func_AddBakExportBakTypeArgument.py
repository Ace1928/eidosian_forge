from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddBakExportBakTypeArgument(parser):
    """Add the 'bak-type' argument to the parser for bak import."""
    choices = [messages.ExportContext.BakExportOptionsValue.BakTypeValueValuesEnum.FULL.name, messages.ExportContext.BakExportOptionsValue.BakTypeValueValuesEnum.DIFF.name]
    help_text = 'Type of bak file that will be exported, FULL or DIFF. SQL Server only.'
    parser.add_argument('--bak-type', choices=choices, required=False, default=messages.ExportContext.BakExportOptionsValue.BakTypeValueValuesEnum.FULL.name, help=help_text)
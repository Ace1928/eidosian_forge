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
def GetInstanceListFormatForNetworkArchitectureUpgrade():
    """Returns the table format for listing instances with current network architecture field."""
    table_format = '{} table({})'.format(INSTANCES_USERLABELS_FORMAT, ','.join(INSTANCES_FORMAT_COLUMNS_WITH_NETWORK_ARCHITECTURE))
    return table_format
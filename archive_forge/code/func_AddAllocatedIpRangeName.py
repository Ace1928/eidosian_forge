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
def AddAllocatedIpRangeName(parser):
    """Adds the `--allocated-ip-range-name` flag to the parser."""
    parser.add_argument('--allocated-ip-range-name', required=False, help="The name of the IP range allocated for a Cloud SQL instance with private network connectivity. For example: 'google-managed-services-default'. If set, the instance IP is created in the allocated range represented by this name.")
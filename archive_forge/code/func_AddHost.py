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
def AddHost(parser):
    """Add the '--host' flag to the parser."""
    parser.add_argument('--host', help="Cloud SQL user's hostname expressed as a specific IP address or address range. `%` denotes an unrestricted hostname. Applicable flag for MySQL instances; ignored for all other engines. Note, if you connect to your instance using IP addresses, you must add your client IP address as an authorized address, even if your hostname is unrestricted. For more information, see [Configure IP](https://cloud.google.com/sql/docs/mysql/configure-ip).")
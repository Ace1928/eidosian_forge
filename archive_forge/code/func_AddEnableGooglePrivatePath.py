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
def AddEnableGooglePrivatePath(parser, show_negated_in_help=False, hidden=False):
    kwargs = _GetKwargsForBoolFlag(show_negated_in_help)
    parser.add_argument('--enable-google-private-path', required=False, help="Enable a private path for Google Cloud services. This flag specifies whether the instance is accessible to internal Google Cloud services such as BigQuery. This is only applicable to MySQL and PostgreSQL instances that don't use public IP. Currently, SQL Server isn't supported.", hidden=hidden, **kwargs)
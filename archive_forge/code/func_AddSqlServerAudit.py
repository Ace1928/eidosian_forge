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
def AddSqlServerAudit(parser, hidden=False):
    """Adds SQL Server audit related flags to the parser."""
    parser.add_argument('--audit-bucket-path', required=False, help='The location, as a Cloud Storage bucket, to which audit files are uploaded. The URI is in the form gs://bucketName/folderName. Only available for SQL Server instances.', hidden=hidden)
    parser.add_argument('--audit-retention-interval', default=None, type=arg_parsers.Duration(upper_bound='7d'), required=False, help='The number of days for audit log retention on disk, for example, 3dfor 3 days. Only available for SQL Server instances.', hidden=hidden)
    parser.add_argument('--audit-upload-interval', default=None, type=arg_parsers.Duration(upper_bound='720m'), required=False, help='How often to upload audit logs (audit files), for example, 30mfor 30 minutes. Only available for SQL Server instances.', hidden=hidden)
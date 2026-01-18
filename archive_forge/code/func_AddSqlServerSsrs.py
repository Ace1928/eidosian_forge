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
def AddSqlServerSsrs(parser):
    """Adds SQL Server Reporting Services (SSRS) related flags to the parser."""
    parser.add_argument('--setup-login', required=True, help='Existing login in the Cloud SQL for SQL Server instance that is used as the setup login for SSRS setup.')
    parser.add_argument('--service-login', required=True, help='Existing login in the Cloud SQL for SQL Server instance that is used as the service login for SSRS setup.')
    parser.add_argument('--report-database', required=True, help='Existing or new report database name in the Cloud SQL for SQL Server instance that is used for SSRS setup.')
    parser.add_argument('--duration', default=None, type=arg_parsers.Duration(lower_bound='1h', upper_bound='12h'), required=False, help='Time duration, in hours, that the lease will be active to allow SSRS setup. Default lease duration is 5 hours if this flag is not specified.')
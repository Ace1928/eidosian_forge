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
def AddSwitchoverDbTimeout(parser):
    parser.add_argument('--db-timeout', default=None, type=arg_parsers.Duration(lower_bound='1s', upper_bound='1d'), required=False, help='(MySQL only) Cloud SQL instance operations timeout, which is the sum of all database operations. Default value is 10 minutes and can be modified to a maximum value of 24h.')
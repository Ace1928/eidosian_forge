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
def AddMasterInstanceName(parser, hidden=False):
    parser.add_argument('--master-instance-name', required=False, hidden=hidden, help='Name of the instance which will act as master in the replication setup. The newly created instance will be a read replica of the specified master instance.')
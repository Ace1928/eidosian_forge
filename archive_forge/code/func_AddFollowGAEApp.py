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
def AddFollowGAEApp(parser):
    parser.add_argument('--follow-gae-app', required=False, help='First Generation instances only. The App Engine app this instance should follow. It must be in the same region as the instance. WARNING: Instance may be restarted.')
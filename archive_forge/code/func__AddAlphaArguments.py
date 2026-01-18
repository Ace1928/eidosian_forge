from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core.util import times
import six
def _AddAlphaArguments(parser, release_track):
    if release_track == base.ReleaseTrack.ALPHA:
        parser.add_argument('--secondary-worker-standard-capacity-base', type=int, help='\n              The number of standard VMs in the Spot and Standard Mix\n        feature.\n              ')
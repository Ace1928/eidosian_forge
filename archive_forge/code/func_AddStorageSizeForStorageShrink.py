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
def AddStorageSizeForStorageShrink(parser):
    parser.add_argument('--storage-size', type=arg_parsers.BinarySize(lower_bound='10GB', upper_bound='65536GB', suggested_binary_size_scales=['GB']), required=True, help='The target storage size must be an integer that represents the number of GB. For example, --storage-size=10GB')
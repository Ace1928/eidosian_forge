from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddManBlocksFlag(parser):
    """Adds --man-blocks flag."""
    parser.add_argument('--man-blocks', type=arg_parsers.ArgList(), metavar='BLOCK', help='Master Authorized Network. Allows users to specify multiple blocks to access the Kubernetescontrol plane from this block. Defaults to `0.0.0.0/0` if flag is not provided.')
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddNodeCount(parser, required=True):
    parser.add_argument('--node-count', required=required, help='\n      Default nodeCount used by this node pool.\n      ')
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddDefaultMaxPodsPerNode(parser):
    parser.add_argument('--default-max-pods-per-node', help='The default maximum number of pods per node.')
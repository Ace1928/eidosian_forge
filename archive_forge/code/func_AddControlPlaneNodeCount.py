from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddControlPlaneNodeCount(parser):
    parser.add_argument('--control-plane-node-count', help='\n      The number of local control plane nodes in a cluster. Use one to create\n      a single-node control plane or use three to create a high availability\n      control plane.\n      Any other numbers of nodes will not be accepted.\n      ')
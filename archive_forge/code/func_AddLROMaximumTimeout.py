from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddLROMaximumTimeout(parser):
    parser.add_argument('--lro-timeout', help='\n      Overwrite the default LRO maximum timeout.\n      ')
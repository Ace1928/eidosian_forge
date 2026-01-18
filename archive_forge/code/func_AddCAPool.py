from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddCAPool(parser, help_text='CA Pool path for private instance.'):
    parser.add_argument('--ca-pool', dest='ca_pool', required=False, help=help_text)
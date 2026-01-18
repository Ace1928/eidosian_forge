from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddIsPrivate(parser, help_text='Bool indicator for private instance.'):
    parser.add_argument('--is-private', dest='is_private', action='store_true', required=False, help=help_text)
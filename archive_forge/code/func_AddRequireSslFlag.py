from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddRequireSslFlag(parser):
    """Adds a --require-ssl flag to the given parser."""
    help_text = 'Whether SSL connections over IP should be enforced or not.'
    parser.add_argument('--require-ssl', help=help_text, action='store_true', dest='require_ssl', default=False)
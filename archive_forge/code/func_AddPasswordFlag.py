from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddPasswordFlag(parser):
    """Add the password field to the parser."""
    parser.add_argument('--password', required=True, help="Initial password for the 'postgres' user.")
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddRootPassword(parser):
    """Add the root password field to the parser."""
    parser.add_argument('--root-password', required=False, help="Root Cloud SQL user's password.")
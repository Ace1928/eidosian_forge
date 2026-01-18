from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddZoneFlag(parser):
    """Adds a --zone flag to the given parser."""
    help_text = '    Google Cloud Platform zone where your Cloud SQL database instance is\n    located.\n  '
    parser.add_argument('--zone', help=help_text)
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddCdnSignedUrlKeyName(parser, required=False):
    """Adds the Cloud CDN Signed URL key name argument to the argparse."""
    parser.add_argument('--key-name', required=required, help='Name of the Cloud CDN Signed URL key.')
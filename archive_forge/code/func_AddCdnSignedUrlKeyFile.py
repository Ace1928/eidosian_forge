from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddCdnSignedUrlKeyFile(parser, required=False):
    """Adds the Cloud CDN Signed URL key file argument to the argparse."""
    parser.add_argument('--key-file', required=required, metavar='LOCAL_FILE_PATH', help='      The file containing the RFC 4648 Section 5 base64url encoded 128-bit\n      secret key for Cloud CDN Signed URL. It is vital that the key is\n      strongly random. One way to generate such a key is with the following\n      command:\n\n          head -c 16 /dev/random | base64 | tr +/ -_ > [KEY_FILE_NAME]\n\n      ')
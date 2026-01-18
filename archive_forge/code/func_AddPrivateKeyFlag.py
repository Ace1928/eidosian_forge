from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
def AddPrivateKeyFlag(parser, required=False):
    """Adds --private-key flag to the given parser."""
    help_text = '    Unencrypted PKCS#1 or PKCS#8 PEM-encoded private key associated with\n    the Client Certificate. Database Migration Service encrypts the value when\n    storing it.\n  '
    parser.add_argument('--private-key', help=help_text, required=required)
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
def AddCaCertificateFlag(parser, required=False):
    """Adds --ca-certificate flag to the given parser."""
    help_text = "    x509 PEM-encoded certificate of the CA that signed the database\n    server's certificate. Database Migration Service will use this certificate to verify\n    it's connecting to the correct host. Database Migration Service encrypts the\n    value when storing it.\n  "
    parser.add_argument('--ca-certificate', help=help_text, required=required)
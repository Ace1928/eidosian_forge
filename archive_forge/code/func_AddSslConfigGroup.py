from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
def AddSslConfigGroup(parser, release_track):
    """Adds ssl server only & server client config group to the given parser."""
    ssl_config = parser.add_group()
    AddCaCertificateFlag(ssl_config, True)
    client_cert = ssl_config.add_group()
    AddPrivateKeyFlag(client_cert, required=True)
    if api_util.GetApiVersion(release_track) == 'v1alpha2':
        AddCertificateFlag(client_cert, required=True)
    else:
        AddClientCertificateFlag(client_cert, required=True)
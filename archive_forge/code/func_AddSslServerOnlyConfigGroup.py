from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
def AddSslServerOnlyConfigGroup(parser):
    """Adds ssl server only config group to the given parser."""
    ssl_config = parser.add_group()
    AddCaCertificateFlag(ssl_config, True)
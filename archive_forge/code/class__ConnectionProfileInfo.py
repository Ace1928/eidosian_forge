from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import connection_profiles
from googlecloudsdk.api_lib.database_migration import resource_args
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
class _ConnectionProfileInfo(object):
    """Container for connection profile data using in list display."""

    def __init__(self, message, host, engine):
        self.display_name = message.displayName
        self.name = message.name
        self.state = message.state
        self.provider_display = message.provider
        self.host = host
        self.create_time = message.createTime
        self.engine = engine
        if message.cloudsql:
            if not message.provider:
                self.provider_display = 'CLOUDSQL'
            self.provider_display = '{}_{}'.format(self.provider_display, 'REPLICA')
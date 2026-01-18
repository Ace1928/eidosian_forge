from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.core import resources
def GetPrivateConnectionURI(resource):
    private_connection = resources.REGISTRY.ParseRelativeName(resource.name, collection='datamigration.projects.locations.privateConnections')
    return private_connection.SelfLink()
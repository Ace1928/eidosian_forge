from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.eventarc import common_publishing
from googlecloudsdk.api_lib.eventarc.base import EventarcClientBase
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import resources
def GetChannelConnectionsURI(resource):
    channel_connections = resources.REGISTRY.ParseRelativeName(resource.name, collection='eventarc.projects.locations.channelConnections')
    return channel_connections.SelfLink()
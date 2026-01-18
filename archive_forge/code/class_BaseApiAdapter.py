from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.scheduler import jobs
from googlecloudsdk.api_lib.scheduler import locations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
class BaseApiAdapter(object):

    def __init__(self, api_version):
        self.client = apis.GetClientInstance(API_NAME, api_version)
        self.messages = self.client.MESSAGES_MODULE
        self.locations = locations.Locations(self.client.MESSAGES_MODULE, self.client.projects_locations)
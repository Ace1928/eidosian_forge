from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import util
class CrawlersClient(object):
    """Cloud Datacatalog crawlers client."""

    def __init__(self):
        self.client = util.GetClientInstance(api_version=_CRAWLER_API_VERSION)
        self.messages = util.GetMessagesModule(api_version=_CRAWLER_API_VERSION)
        self.service = self.client.projects_crawlers

    def Get(self, crawler_ref):
        request = self.messages.DatacatalogProjectsCrawlersGetRequest(name=crawler_ref.RelativeName())
        return self.service.Get(request)
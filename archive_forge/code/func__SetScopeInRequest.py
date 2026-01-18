from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import crawlers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
def _SetScopeInRequest(crawl_scope, buckets, request, messages):
    """Returns request with the crawl scope set."""
    if crawl_scope == 'bucket':
        if not buckets:
            raise InvalidCrawlScopeError('At least one bucket must be included in the crawl scope of a bucket-scoped crawler.')
        arg_utils.SetFieldInMessage(request, 'googleCloudDatacatalogV1alpha3Crawler.config.bucketScope.buckets', buckets)
    elif crawl_scope == 'project':
        arg_utils.SetFieldInMessage(request, 'googleCloudDatacatalogV1alpha3Crawler.config.projectScope', messages.GoogleCloudDatacatalogV1alpha3ParentProjectScope())
    elif crawl_scope == 'organization':
        arg_utils.SetFieldInMessage(request, 'googleCloudDatacatalogV1alpha3Crawler.config.organizationScope', messages.GoogleCloudDatacatalogV1alpha3ParentOrganizationScope())
    return request
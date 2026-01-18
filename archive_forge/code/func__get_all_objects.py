import xml.sax
import time
import boto
from boto.connection import AWSAuthConnection
from boto import handler
from boto.cloudfront.distribution import Distribution, DistributionSummary, DistributionConfig
from boto.cloudfront.distribution import StreamingDistribution, StreamingDistributionSummary, StreamingDistributionConfig
from boto.cloudfront.identity import OriginAccessIdentity
from boto.cloudfront.identity import OriginAccessIdentitySummary
from boto.cloudfront.identity import OriginAccessIdentityConfig
from boto.cloudfront.invalidation import InvalidationBatch, InvalidationSummary, InvalidationListResultSet
from boto.resultset import ResultSet
from boto.cloudfront.exception import CloudFrontServerError
def _get_all_objects(self, resource, tags, result_set_class=None, result_set_kwargs=None):
    if not tags:
        tags = [('DistributionSummary', DistributionSummary)]
    response = self.make_request('GET', '/%s/%s' % (self.Version, resource))
    body = response.read()
    boto.log.debug(body)
    if response.status >= 300:
        raise CloudFrontServerError(response.status, response.reason, body)
    rs_class = result_set_class or ResultSet
    rs_kwargs = result_set_kwargs or dict()
    rs = rs_class(tags, **rs_kwargs)
    h = handler.XmlHandler(rs, self)
    xml.sax.parseString(body, h)
    return rs
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
def _create_object(self, config, resource, dist_class):
    response = self.make_request('POST', '/%s/%s' % (self.Version, resource), {'Content-Type': 'text/xml'}, data=config.to_xml())
    body = response.read()
    boto.log.debug(body)
    if response.status == 201:
        d = dist_class(connection=self)
        h = handler.XmlHandler(d, self)
        xml.sax.parseString(body, h)
        d.etag = self.get_etag(response)
        return d
    else:
        raise CloudFrontServerError(response.status, response.reason, body)
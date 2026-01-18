import uuid
import base64
import time
from boto.compat import six, json
from boto.cloudfront.identity import OriginAccessIdentity
from boto.cloudfront.object import Object, StreamingObject
from boto.cloudfront.signers import ActiveTrustedSigners, TrustedSigners
from boto.cloudfront.logging import LoggingInfo
from boto.cloudfront.origin import S3Origin, CustomOrigin
from boto.s3.acl import ACL
class DistributionSummary(object):

    def __init__(self, connection=None, domain_name='', id='', last_modified_time=None, status='', origin=None, cname='', comment='', enabled=False):
        self.connection = connection
        self.domain_name = domain_name
        self.id = id
        self.last_modified_time = last_modified_time
        self.status = status
        self.origin = origin
        self.enabled = enabled
        self.cnames = []
        if cname:
            self.cnames.append(cname)
        self.comment = comment
        self.trusted_signers = None
        self.etag = None
        self.streaming = False

    def __repr__(self):
        return 'DistributionSummary:%s' % self.domain_name

    def startElement(self, name, attrs, connection):
        if name == 'TrustedSigners':
            self.trusted_signers = TrustedSigners()
            return self.trusted_signers
        elif name == 'S3Origin':
            self.origin = S3Origin()
            return self.origin
        elif name == 'CustomOrigin':
            self.origin = CustomOrigin()
            return self.origin
        return None

    def endElement(self, name, value, connection):
        if name == 'Id':
            self.id = value
        elif name == 'Status':
            self.status = value
        elif name == 'LastModifiedTime':
            self.last_modified_time = value
        elif name == 'DomainName':
            self.domain_name = value
        elif name == 'Origin':
            self.origin = value
        elif name == 'CNAME':
            self.cnames.append(value)
        elif name == 'Comment':
            self.comment = value
        elif name == 'Enabled':
            if value.lower() == 'true':
                self.enabled = True
            else:
                self.enabled = False
        elif name == 'StreamingDistributionSummary':
            self.streaming = True
        else:
            setattr(self, name, value)

    def get_distribution(self):
        return self.connection.get_distribution_info(self.id)
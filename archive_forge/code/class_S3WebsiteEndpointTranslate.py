from __future__ import division
import boto
from boto import handler
from boto.resultset import ResultSet
from boto.exception import BotoClientError
from boto.s3.acl import Policy, CannedACLStrings, Grant
from boto.s3.key import Key
from boto.s3.prefix import Prefix
from boto.s3.deletemarker import DeleteMarker
from boto.s3.multipart import MultiPartUpload
from boto.s3.multipart import CompleteMultiPartUpload
from boto.s3.multidelete import MultiDeleteResult
from boto.s3.multidelete import Error
from boto.s3.bucketlistresultset import BucketListResultSet
from boto.s3.bucketlistresultset import VersionedBucketListResultSet
from boto.s3.bucketlistresultset import MultiPartUploadListResultSet
from boto.s3.lifecycle import Lifecycle
from boto.s3.tagging import Tags
from boto.s3.cors import CORSConfiguration
from boto.s3.bucketlogging import BucketLogging
from boto.s3 import website
import boto.jsonresponse
import boto.utils
import xml.sax
import xml.sax.saxutils
import re
import base64
from collections import defaultdict
from boto.compat import BytesIO, six, StringIO, urllib
from boto.utils import get_utf8able_str
class S3WebsiteEndpointTranslate(object):
    trans_region = defaultdict(lambda: 's3-website-us-east-1')
    trans_region['eu-west-1'] = 's3-website-eu-west-1'
    trans_region['eu-central-1'] = 's3-website.eu-central-1'
    trans_region['us-west-1'] = 's3-website-us-west-1'
    trans_region['us-west-2'] = 's3-website-us-west-2'
    trans_region['sa-east-1'] = 's3-website-sa-east-1'
    trans_region['ap-northeast-1'] = 's3-website-ap-northeast-1'
    trans_region['ap-southeast-1'] = 's3-website-ap-southeast-1'
    trans_region['ap-southeast-2'] = 's3-website-ap-southeast-2'
    trans_region['cn-north-1'] = 's3-website.cn-north-1'

    @classmethod
    def translate_region(self, reg):
        return self.trans_region[reg]
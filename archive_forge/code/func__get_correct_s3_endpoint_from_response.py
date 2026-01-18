from datetime import datetime
import errno
import os
import random
import re
import socket
import sys
import time
import xml.sax
import copy
from boto import auth
from boto import auth_handler
import boto
import boto.utils
import boto.handler
import boto.cacerts
from boto import config, UserAgent
from boto.compat import six, http_client, urlparse, quote, encodebytes
from boto.exception import AWSConnectionError
from boto.exception import BotoClientError
from boto.exception import BotoServerError
from boto.exception import PleaseRetryException
from boto.exception import S3ResponseError
from boto.provider import Provider
from boto.resultset import ResultSet
def _get_correct_s3_endpoint_from_response(self, request, err, get_header):
    """Attempt to return a new s3 endpoint using the correct region to
        access a bucket. Return None if a retry is not possible."""
    if callable(get_header):
        region = get_header('x-amz-bucket-region')
        if region:
            boto.log.debug('Got correct region from response headers.')
            return self._fix_s3_endpoint_region(request.host, region)
    if err.region:
        boto.log.debug('Got correct region from parsed xml in err.region.')
        return self._fix_s3_endpoint_region(request.host, err.region)
    elif err.error_code == 'IllegalLocationConstraintException':
        region_regex = 'The (.*?) location constraint is incompatible for the region specific endpoint this request was sent to\\.'
        match = re.search(region_regex, err.body)
        if match and match.group(1) != 'unspecified':
            region = match.group(1)
            boto.log.debug('Got correct region from response body.')
            return self._fix_s3_endpoint_region(request.host, region)
    elif err.endpoint:
        boto.log.debug('Got correct endpoint from response body.')
        return err.endpoint
    boto.log.debug('Sending a bucket HEAD request to get correct region.')
    req = self.build_base_http_request('HEAD', '/', '/', {}, None, '', request.host)
    bucket_head_response = self._mexe(req, None, None)
    region = bucket_head_response.getheader('x-amz-bucket-region')
    if region:
        boto.log.debug('Got correct region from a bucket head request.')
        return self._fix_s3_endpoint_region(request.host, region)
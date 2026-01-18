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
class HTTPResponse(http_client.HTTPResponse):

    def __init__(self, *args, **kwargs):
        http_client.HTTPResponse.__init__(self, *args, **kwargs)
        self._cached_response = ''

    def read(self, amt=None):
        """Read the response.

        This method does not have the same behavior as
        http_client.HTTPResponse.read.  Instead, if this method is called with
        no ``amt`` arg, then the response body will be cached.  Subsequent
        calls to ``read()`` with no args **will return the cached response**.

        """
        if amt is None:
            if not self._cached_response:
                self._cached_response = http_client.HTTPResponse.read(self)
            return self._cached_response
        else:
            return http_client.HTTPResponse.read(self, amt)
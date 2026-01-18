import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib, urlparse, parse_qsl
from libcloud.common.types import MalformedResponseError
from libcloud.common.cloudstack import CloudStackConnection
def _check_request(self, url):
    url = urlparse.urlparse(url)
    query = dict(parse_qsl(url.query))
    self.assertTrue('apiKey' in query)
    self.assertTrue('command' in query)
    self.assertTrue('response' in query)
    self.assertTrue('signature' in query)
    self.assertTrue(query['response'] == 'json')
    return query
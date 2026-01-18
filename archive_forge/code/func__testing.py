import sys
import string
import unittest
from unittest.mock import Mock, patch
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.nfsn import NFSNConnection
def _testing(self, method, url, body, headers):
    if headers['X-NFSN-Authentication'] == mock_header:
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])
    else:
        return (httplib.UNAUTHORIZED, '', {}, httplib.responses[httplib.UNAUTHORIZED])
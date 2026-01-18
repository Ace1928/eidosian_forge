import os
import ssl
import sys
import socket
from unittest import mock
from unittest.mock import Mock, patch
import requests_mock
from requests.exceptions import ConnectTimeout
import libcloud.common.base
from libcloud.http import LibcloudConnection, SignedHTTPSAdapter, LibcloudBaseConnection
from libcloud.test import unittest, no_internet
from libcloud.utils.py3 import assertRaisesRegex
from libcloud.common.base import Response, Connection, CertificateConnection
from libcloud.utils.retry import RETRY_EXCEPTIONS, Retry, RetryForeverOnRateLimitError
from libcloud.common.exceptions import RateLimitReachedError
class ThrowingResponse(Response):

    def __init__(self, *_, **__):
        super().__init__(mock.MagicMock(), mock.MagicMock())

    def parse_body(self):
        return super().parse_body()

    def parse_error(self):
        raise RateLimitReachedError()

    def success(self):
        return False
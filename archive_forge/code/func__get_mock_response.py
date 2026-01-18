import os
import sys
import zlib
from io import StringIO
from unittest import mock
import requests_mock
import libcloud
from libcloud.http import LibcloudConnection
from libcloud.test import unittest
from libcloud.common.base import Connection
from libcloud.utils.loggingconnection import LoggingConnection
def _get_mock_response(self, content_type, body):
    header = mock.Mock()
    header.title.return_value = 'Content-Type'
    header.lower.return_value = 'content-type'
    r = mock.Mock()
    r.version = 11
    r.status = '200'
    r.reason = 'OK'
    r.getheaders.return_value = [(header, content_type)]
    r.read.return_value = body
    return r
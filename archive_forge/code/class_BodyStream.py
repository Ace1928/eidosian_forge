import os
import random
import unittest
import requests
import requests_mock
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import PY2, httplib, parse_qs, urlparse, urlquote, parse_qsl
from libcloud.common.base import Response
class BodyStream(StringIO):

    def next(self, chunk_size=None):
        return StringIO.next(self)

    def __next__(self, chunk_size=None):
        return StringIO.__next__(self)

    def read(self, chunk_size=None):
        return StringIO.read(self)
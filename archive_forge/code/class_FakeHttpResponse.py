import collections
import http.client as http
import io
from unittest import mock
import copy
import os
import sys
import uuid
import fixtures
from oslo_serialization import jsonutils
import webob
from glance.cmd import replicator as glance_replicator
from glance.common import exception
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
class FakeHttpResponse(object):

    def __init__(self, headers, data):
        self.headers = headers
        self.data = io.BytesIO(data)

    def getheaders(self):
        return self.headers

    def read(self, amt=None):
        return self.data.read(amt)
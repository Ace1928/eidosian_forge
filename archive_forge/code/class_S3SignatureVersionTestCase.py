import copy
import pickle
import os
from tests.compat import unittest, mock
from tests.unit import MockServiceWithConfigTestCase
from nose.tools import assert_equal
from boto.auth import HmacAuthV4Handler
from boto.auth import S3HmacAuthV4Handler
from boto.auth import detect_potential_s3sigv4
from boto.auth import detect_potential_sigv4
from boto.connection import HTTPRequest
from boto.provider import Provider
from boto.regioninfo import RegionInfo
class S3SignatureVersionTestCase(object):

    def __init__(self, host, expected_signture_version, anon=None):
        self.host = host
        self.connection = FakeS3Connection(host=host, anon=anon)
        self.expected_signature_version = expected_signture_version

    def run(self):
        auth = self.connection._required_auth_capability()
        message = "Expected signature version ['%s'] for host %s but found %s." % (self.expected_signature_version, self.host, auth)
        assert_equal(auth, [self.expected_signature_version], message)
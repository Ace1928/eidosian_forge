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
class FakeS3Connection(object):

    def __init__(self, *args, **kwargs):
        self.host = kwargs.pop('host', None)
        self.anon = kwargs.pop('anon', None)

    @detect_potential_s3sigv4
    def _required_auth_capability(self):
        if self.anon:
            return ['anon']
        return ['nope']

    def _mexe(self, *args, **kwargs):
        pass
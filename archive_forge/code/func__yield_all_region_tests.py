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
def _yield_all_region_tests(region, expected_signature_version='hmac-v4-s3', dns_suffix='.amazonaws.com'):
    """Yield tests for every variation of a region's endpoints."""
    host = 's3.' + region + dns_suffix
    case = S3SignatureVersionTestCase(host, expected_signature_version)
    yield case.run
    host = 's3-' + region + dns_suffix
    case = S3SignatureVersionTestCase(host, expected_signature_version)
    yield case.run
    host = 'mybucket.s3-' + region + dns_suffix
    case = S3SignatureVersionTestCase(host, expected_signature_version)
    yield case.run
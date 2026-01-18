from hashlib import sha1
import random
import string
import tempfile
import time
from unittest import mock
import requests_mock
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack.object_store.v1 import account
from openstack.object_store.v1 import container
from openstack.object_store.v1 import obj
from openstack.tests.unit.cloud import test_object as base_test_object
from openstack.tests.unit import test_proxy_base
class TestTempURLBytesPathUnicodeKey(TestTempURL):
    url = u'/v1/ä/c/ó'.encode('utf-8')
    key = u'kéy'
    expected_url = url + b'?temp_url_sig=temp_url_signature&temp_url_expires=1400003600'
    expected_body = b'\n'.join([b'GET', b'1400003600', url])
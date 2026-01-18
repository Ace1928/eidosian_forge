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
class TestTempURLUnicodePathBytesKey(TestTempURL):
    url = u'/v1/ä/c/ó'
    key = u'kéy'.encode('utf-8')
    expected_url = u'%s?temp_url_sig=temp_url_signature&temp_url_expires=1400003600' % url
    expected_body = '\n'.join([u'GET', u'1400003600', url]).encode('utf-8')
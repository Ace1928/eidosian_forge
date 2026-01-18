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
def _test_object_delete(self, ignore):
    expected_kwargs = {'ignore_missing': ignore, 'container': 'name'}
    self._verify('openstack.proxy.Proxy._delete', self.proxy.delete_object, method_args=['resource'], method_kwargs=expected_kwargs, expected_args=[obj.Object, 'resource'], expected_kwargs=expected_kwargs)
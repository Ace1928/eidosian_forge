import os
import uuid
from oslo_config import cfg
from oslo_utils import uuidutils
from oslotest import base
import requests
from testtools import testcase
from castellan.common import exception
from castellan.key_manager import vault_key_manager
from castellan.tests.functional import config
from castellan.tests.functional.key_manager import test_key_manager
def _create_policy(self, vault_policy):
    params = {'rules': TEST_POLICY.format(backend=self.mountpoint)}
    self.session.put('{}/{}'.format(self.vault_url, POLICY_ENDPOINT.format(policy_name=vault_policy)), json=params)
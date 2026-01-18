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
def _retrieve_approle(self, vault_approle):
    approle_role_id = self.session.get('{}/v1/auth/approle/role/{}/role-id'.format(self.vault_url, vault_approle)).json()['data']['role_id']
    approle_secret_id = self.session.post('{}/v1/auth/approle/role/{}/secret-id'.format(self.vault_url, vault_approle)).json()['data']['secret_id']
    return (approle_role_id, approle_secret_id)
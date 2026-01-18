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
class VaultKeyManagerAltMountpointTestCase(VaultKeyManagerAppRoleTestCase):
    mountpoint = 'different-secrets'
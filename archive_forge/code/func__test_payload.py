import base64
import datetime
import hashlib
import os
from unittest import mock
import uuid
from oslo_utils import timeutils
from keystone.common import fernet_utils
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.receipt.providers import fernet
from keystone.receipt import receipt_formatters
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.token import provider as token_provider
def _test_payload(self, payload_class, exp_user_id=None, exp_methods=None):
    exp_user_id = exp_user_id or uuid.uuid4().hex
    exp_methods = exp_methods or ['password']
    exp_expires_at = utils.isotime(timeutils.utcnow(), subsecond=True)
    payload = payload_class.assemble(exp_user_id, exp_methods, exp_expires_at)
    user_id, methods, expires_at = payload_class.disassemble(payload)
    self.assertEqual(exp_user_id, user_id)
    self.assertEqual(exp_methods, methods)
    self.assertTimestampsEqual(exp_expires_at, expires_at)
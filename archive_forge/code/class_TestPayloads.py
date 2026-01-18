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
class TestPayloads(unit.TestCase):

    def setUp(self):
        super(TestPayloads, self).setUp()
        self.useFixture(ksfixtures.ConfigAuthPlugins(self.config_fixture, ['totp', 'token', 'password']))

    def assertTimestampsEqual(self, expected, actual):
        exp_time = timeutils.parse_isotime(expected)
        actual_time = timeutils.parse_isotime(actual)
        return self.assertCloseEnoughForGovernmentWork(exp_time, actual_time, delta=1e-05)

    def test_strings_can_be_converted_to_bytes(self):
        s = token_provider.random_urlsafe_str()
        self.assertIsInstance(s, str)
        b = receipt_formatters.ReceiptPayload.random_urlsafe_str_to_bytes(s)
        self.assertIsInstance(b, bytes)

    def test_uuid_hex_to_byte_conversions(self):
        payload_cls = receipt_formatters.ReceiptPayload
        expected_hex_uuid = uuid.uuid4().hex
        uuid_obj = uuid.UUID(expected_hex_uuid)
        expected_uuid_in_bytes = uuid_obj.bytes
        actual_uuid_in_bytes = payload_cls.convert_uuid_hex_to_bytes(expected_hex_uuid)
        self.assertEqual(expected_uuid_in_bytes, actual_uuid_in_bytes)
        actual_hex_uuid = payload_cls.convert_uuid_bytes_to_hex(expected_uuid_in_bytes)
        self.assertEqual(expected_hex_uuid, actual_hex_uuid)

    def test_time_string_to_float_conversions(self):
        payload_cls = receipt_formatters.ReceiptPayload
        original_time_str = utils.isotime(subsecond=True)
        time_obj = timeutils.parse_isotime(original_time_str)
        expected_time_float = (timeutils.normalize_time(time_obj) - datetime.datetime.utcfromtimestamp(0)).total_seconds()
        self.assertIsInstance(expected_time_float, float)
        actual_time_float = payload_cls._convert_time_string_to_float(original_time_str)
        self.assertIsInstance(actual_time_float, float)
        self.assertEqual(expected_time_float, actual_time_float)
        time_object = datetime.datetime.utcfromtimestamp(actual_time_float)
        expected_time_str = utils.isotime(time_object, subsecond=True)
        actual_time_str = payload_cls._convert_float_to_time_string(actual_time_float)
        self.assertEqual(expected_time_str, actual_time_str)

    def _test_payload(self, payload_class, exp_user_id=None, exp_methods=None):
        exp_user_id = exp_user_id or uuid.uuid4().hex
        exp_methods = exp_methods or ['password']
        exp_expires_at = utils.isotime(timeutils.utcnow(), subsecond=True)
        payload = payload_class.assemble(exp_user_id, exp_methods, exp_expires_at)
        user_id, methods, expires_at = payload_class.disassemble(payload)
        self.assertEqual(exp_user_id, user_id)
        self.assertEqual(exp_methods, methods)
        self.assertTimestampsEqual(exp_expires_at, expires_at)

    def test_payload(self):
        self._test_payload(receipt_formatters.ReceiptPayload)

    def test_payload_multiple_methods(self):
        self._test_payload(receipt_formatters.ReceiptPayload, exp_methods=['password', 'totp'])
import datetime
from unittest import mock
import glance_store
from oslo_config import cfg
import oslo_messaging
import webob
import glance.async_
from glance.common import exception
from glance.common import timeutils
import glance.context
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
from glance.tests import utils
def _assert_image_member_with_notifier(self, output_log, deleted=False):
    self.assertEqual(self.image_member.member_id, output_log['payload']['member_id'])
    self.assertEqual(self.image_member.image_id, output_log['payload']['image_id'])
    self.assertEqual(self.image_member.status, output_log['payload']['status'])
    self.assertEqual(timeutils.isotime(self.image_member.created_at), output_log['payload']['created_at'])
    self.assertEqual(timeutils.isotime(self.image_member.updated_at), output_log['payload']['updated_at'])
    if deleted:
        self.assertTrue(output_log['payload']['deleted'])
        self.assertIsNotNone(output_log['payload']['deleted_at'])
    else:
        self.assertFalse(output_log['payload']['deleted'])
        self.assertIsNone(output_log['payload']['deleted_at'])
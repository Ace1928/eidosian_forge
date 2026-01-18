from unittest import mock
import stevedore
import testtools
from oslo_messaging import server
from oslo_messaging.tests import utils as test_utils
def _test_list_opts(self, result):
    self.assertEqual(5, len(result))
    groups = [g for g, l in result]
    self.assertIn(None, groups)
    self.assertIn('oslo_messaging_amqp', groups)
    self.assertIn('oslo_messaging_notifications', groups)
    self.assertIn('oslo_messaging_rabbit', groups)
    self.assertIn('oslo_messaging_kafka', groups)
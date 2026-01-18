from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
class TestRequestContextSerializer(base.BaseTestCase):

    def setUp(self):
        super(TestRequestContextSerializer, self).setUp()
        self.mock_base = mock.Mock()
        self.ser = rpc.RequestContextSerializer(self.mock_base)
        self.ser_null = rpc.RequestContextSerializer(None)

    def test_serialize_entity(self):
        self.mock_base.serialize_entity.return_value = 'foo'
        ser_ent = self.ser.serialize_entity('context', 'entity')
        self.mock_base.serialize_entity.assert_called_once_with('context', 'entity')
        self.assertEqual('foo', ser_ent)

    def test_deserialize_entity(self):
        self.mock_base.deserialize_entity.return_value = 'foo'
        deser_ent = self.ser.deserialize_entity('context', 'entity')
        self.mock_base.deserialize_entity.assert_called_once_with('context', 'entity')
        self.assertEqual('foo', deser_ent)

    def test_deserialize_entity_null_base(self):
        deser_ent = self.ser_null.deserialize_entity('context', 'entity')
        self.assertEqual('entity', deser_ent)

    def test_serialize_context(self):
        context = mock.Mock()
        self.ser.serialize_context(context)
        context.to_dict.assert_called_once_with()

    def test_deserialize_context(self):
        context_dict = {'foo': 'bar', 'user_id': 1, 'tenant_id': 1, 'is_admin': True}
        c = self.ser.deserialize_context(context_dict)
        self.assertEqual(1, c.user_id)
        self.assertEqual(1, c.project_id)

    def test_deserialize_context_no_user_id(self):
        context_dict = {'foo': 'bar', 'user': 1, 'tenant_id': 1, 'is_admin': True}
        c = self.ser.deserialize_context(context_dict)
        self.assertEqual(1, c.user_id)
        self.assertEqual(1, c.project_id)

    def test_deserialize_context_no_tenant_id(self):
        context_dict = {'foo': 'bar', 'user_id': 1, 'project_id': 1, 'is_admin': True}
        c = self.ser.deserialize_context(context_dict)
        self.assertEqual(1, c.user_id)
        self.assertEqual(1, c.project_id)

    def test_deserialize_context_no_ids(self):
        context_dict = {'foo': 'bar', 'is_admin': True}
        c = self.ser.deserialize_context(context_dict)
        self.assertIsNone(c.user_id)
        self.assertIsNone(c.project_id)
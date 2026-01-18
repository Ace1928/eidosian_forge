from unittest import mock
import uuid
from oslo_config import cfg
from troveclient import exceptions as troveexc
from troveclient.v1 import users
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import trove
from heat.engine import resource
from heat.engine.resources.openstack.trove import instance as dbinstance
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
@mock.patch.object(resource.Resource, 'client_plugin')
@mock.patch.object(resource.Resource, 'client')
class InstanceUpdateTests(common.HeatTestCase):

    def setUp(self):
        super(InstanceUpdateTests, self).setUp()
        self._stack = utils.parse_stack(template_format.parse(db_template))
        testprops = {'name': 'testinstance', 'flavor': 'foo', 'datastore_type': 'database', 'datastore_version': '1', 'size': 10, 'databases': [{'name': 'bar'}, {'name': 'biff'}], 'users': [{'name': 'baz', 'password': 'password', 'databases': ['bar']}, {'name': 'deleted', 'password': 'password', 'databases': ['biff']}]}
        self._rdef = rsrc_defn.ResourceDefinition('test', dbinstance.Instance, properties=testprops)

    def test_handle_no_update(self, mock_client, mock_plugin):
        trove = dbinstance.Instance('test', self._rdef, self._stack)
        self.assertEqual({}, trove.handle_update(None, None, {}))

    def test_handle_update_name(self, mock_client, mock_plugin):
        prop_diff = {'name': 'changed'}
        trove = dbinstance.Instance('test', self._rdef, self._stack)
        self.assertEqual(prop_diff, trove.handle_update(None, None, prop_diff))

    def test_handle_update_databases(self, mock_client, mock_plugin):
        prop_diff = {'databases': [{'name': 'bar', 'character_set': 'ascii'}, {'name': 'baz'}]}
        mget = mock_client().databases.list
        mbar = mock.Mock(name='bar')
        mbar.name = 'bar'
        mbiff = mock.Mock(name='biff')
        mbiff.name = 'biff'
        mget.return_value = [mbar, mbiff]
        trove = dbinstance.Instance('test', self._rdef, self._stack)
        expected = {'databases': [{'character_set': 'ascii', 'name': 'bar'}, {'ACTION': 'CREATE', 'name': 'baz'}, {'ACTION': 'DELETE', 'name': 'biff'}]}
        self.assertEqual(expected, trove.handle_update(None, None, prop_diff))

    def test_handle_update_users(self, mock_client, mock_plugin):
        prop_diff = {'users': [{'name': 'baz', 'password': 'changed', 'databases': ['bar', 'biff']}, {'name': 'user2', 'password': 'password', 'databases': ['biff', 'bar']}]}
        uget = mock_client().users
        mbaz = mock.Mock(name='baz')
        mbaz.name = 'baz'
        mdel = mock.Mock(name='deleted')
        mdel.name = 'deleted'
        uget.list.return_value = [mbaz, mdel]
        trove = dbinstance.Instance('test', self._rdef, self._stack)
        expected = {'users': [{'databases': ['bar', 'biff'], 'name': 'baz', 'password': 'changed'}, {'ACTION': 'CREATE', 'databases': ['biff', 'bar'], 'name': 'user2', 'password': 'password'}, {'ACTION': 'DELETE', 'name': 'deleted'}]}
        self.assertEqual(expected, trove.handle_update(None, None, prop_diff))

    def test_handle_update_flavor(self, mock_client, mock_plugin):
        prop_diff = {'flavor': 1234}
        trove = dbinstance.Instance('test', self._rdef, self._stack)
        expected = {'flavor': 1234}
        self.assertEqual(expected, trove.handle_update(None, None, prop_diff))

    def test_handle_update_size(self, mock_client, mock_plugin):
        prop_diff = {'size': 42}
        trove = dbinstance.Instance('test', self._rdef, self._stack)
        expected = {'size': 42}
        self.assertEqual(expected, trove.handle_update(None, None, prop_diff))

    def test_check_complete_none(self, mock_client, mock_plugin):
        trove = dbinstance.Instance('test', self._rdef, self._stack)
        self.assertTrue(trove.check_update_complete({}))

    def test_check_complete_error(self, mock_client, mock_plugin):
        mock_instance = mock.Mock(status='ERROR')
        mock_client().instances.get.return_value = mock_instance
        trove = dbinstance.Instance('test', self._rdef, self._stack)
        exc = self.assertRaises(exception.ResourceInError, trove.check_update_complete, {'foo': 'bar'})
        msg = 'The last operation for the database instance failed'
        self.assertIn(msg, str(exc))

    def test_check_client_exceptions(self, mock_client, mock_plugin):
        mock_instance = mock.Mock(status='ACTIVE')
        mock_client().instances.get.return_value = mock_instance
        mock_plugin().is_client_exception.return_value = True
        mock_plugin().is_over_limit.side_effect = [True, False]
        trove = dbinstance.Instance('test', self._rdef, self._stack)
        with mock.patch.object(trove, '_update_flavor') as mupdate:
            mupdate.side_effect = [Exception('test'), Exception("No change was requested because I'm testing")]
            self.assertFalse(trove.check_update_complete({'foo': 'bar'}))
            self.assertFalse(trove.check_update_complete({'foo': 'bar'}))
            self.assertEqual(2, mupdate.call_count)
            self.assertEqual(2, mock_plugin().is_client_exception.call_count)

    def test_check_complete_status(self, mock_client, mock_plugin):
        mock_instance = mock.Mock(status='RESIZING')
        mock_client().instances.get.return_value = mock_instance
        updates = {'foo': 'bar'}
        trove = dbinstance.Instance('test', self._rdef, self._stack)
        self.assertFalse(trove.check_update_complete(updates))

    def test_check_complete_name(self, mock_client, mock_plugin):
        mock_instance = mock.Mock(status='ACTIVE', name='mock_instance')
        mock_client().instances.get.return_value = mock_instance
        updates = {'name': 'changed'}
        trove = dbinstance.Instance('test', self._rdef, self._stack)
        self.assertFalse(trove.check_update_complete(updates))
        mock_instance.name = 'changed'
        self.assertTrue(trove.check_update_complete(updates))
        mock_client().instances.edit.assert_called_once_with(mock_instance, name='changed')

    def test_check_complete_databases(self, mock_client, mock_plugin):
        mock_instance = mock.Mock(status='ACTIVE', name='mock_instance')
        mock_client().instances.get.return_value = mock_instance
        updates = {'databases': [{'name': 'bar', 'character_set': 'ascii'}, {'ACTION': 'CREATE', 'name': 'baz'}, {'ACTION': 'DELETE', 'name': 'biff'}]}
        trove = dbinstance.Instance('test', self._rdef, self._stack)
        self.assertTrue(trove.check_update_complete(updates))
        mcreate = mock_client().databases.create
        mdelete = mock_client().databases.delete
        mcreate.assert_called_once_with(mock_instance, [{'name': 'baz'}])
        mdelete.assert_called_once_with(mock_instance, 'biff')

    def test_check_complete_users(self, mock_client, mock_plugin):
        mock_instance = mock.Mock(status='ACTIVE', name='mock_instance')
        mock_client().instances.get.return_value = mock_instance
        mock_plugin().is_client_exception.return_value = False
        mock_client().users.get.return_value = users.User(None, {'databases': [{'name': 'bar'}, {'name': 'buzz'}], 'name': 'baz'}, loaded=True)
        updates = {'users': [{'databases': ['bar', 'biff'], 'name': 'baz', 'password': 'changed'}, {'ACTION': 'CREATE', 'databases': ['biff', 'bar'], 'name': 'user2', 'password': 'password'}, {'ACTION': 'DELETE', 'name': 'deleted'}]}
        trove = dbinstance.Instance('test', self._rdef, self._stack)
        self.assertTrue(trove.check_update_complete(updates))
        create_calls = [mock.call(mock_instance, [{'password': 'password', 'databases': [{'name': 'biff'}, {'name': 'bar'}], 'name': 'user2'}])]
        delete_calls = [mock.call(mock_instance, 'deleted')]
        mock_client().users.create.assert_has_calls(create_calls)
        mock_client().users.delete.assert_has_calls(delete_calls)
        self.assertEqual(1, mock_client().users.create.call_count)
        self.assertEqual(1, mock_client().users.delete.call_count)
        updateattr = mock_client().users.update_attributes
        updateattr.assert_called_once_with(mock_instance, 'baz', newuserattr={'password': 'changed'}, hostname=mock.ANY)
        mock_client().users.grant.assert_called_once_with(mock_instance, 'baz', ['biff'])
        mock_client().users.revoke.assert_called_once_with(mock_instance, 'baz', ['buzz'])

    def test_check_complete_flavor(self, mock_client, mock_plugin):
        mock_instance = mock.Mock(status='ACTIVE', flavor={'id': 4567}, name='mock_instance')
        mock_client().instances.get.return_value = mock_instance
        updates = {'flavor': 1234}
        trove = dbinstance.Instance('test', self._rdef, self._stack)
        self.assertFalse(trove.check_update_complete(updates))
        mock_instance.status = 'RESIZING'
        self.assertFalse(trove.check_update_complete(updates))
        mock_instance.status = 'ACTIVE'
        mock_instance.flavor = {'id': 1234}
        self.assertTrue(trove.check_update_complete(updates))

    def test_check_complete_size(self, mock_client, mock_plugin):
        mock_instance = mock.Mock(status='ACTIVE', volume={'size': 24}, name='mock_instance')
        mock_client().instances.get.return_value = mock_instance
        updates = {'size': 42}
        trove = dbinstance.Instance('test', self._rdef, self._stack)
        self.assertFalse(trove.check_update_complete(updates))
        mock_instance.status = 'RESIZING'
        self.assertFalse(trove.check_update_complete(updates))
        mock_instance.status = 'ACTIVE'
        mock_instance.volume = {'size': 42}
        self.assertTrue(trove.check_update_complete(updates))
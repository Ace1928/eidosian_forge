import datetime
import json
import time
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_utils import timeutils
from sqlalchemy import orm
from sqlalchemy.orm import exc
from sqlalchemy.orm import session
from heat.common import context
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.db import api as db_api
from heat.db import models
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import resource as rsrc
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.engine import template_files
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class SqlAlchemyTest(common.HeatTestCase):

    def setUp(self):
        super(SqlAlchemyTest, self).setUp()
        self.fc = fakes_nova.FakeClient()
        self.ctx = utils.dummy_context()

    def _mock_get_image_id_success(self, imageId_input, imageId):
        self.patchobject(glance.GlanceClientPlugin, 'find_image_by_name_or_id', return_value=imageId)

    def _setup_test_stack(self, stack_name, stack_id=None, owner_id=None, stack_user_project_id=None, backup=False):
        t = template_format.parse(wp_template)
        template = tmpl.Template(t, env=environment.Environment({'KeyName': 'test'}))
        stack_id = stack_id or str(uuid.uuid4())
        stack = parser.Stack(self.ctx, stack_name, template, owner_id=owner_id, stack_user_project_id=stack_user_project_id)
        with utils.UUIDStub(stack_id):
            stack.store(backup=backup)
        return (template, stack)

    def _mock_create(self):
        self.patchobject(nova.NovaClientPlugin, 'client', return_value=self.fc)
        self._mock_get_image_id_success('F17-x86_64-gold', 744)
        self.fc.servers.create = mock.Mock(return_value=self.fc.servers.list()[4])
        return self.fc

    def _mock_delete(self):
        self.patchobject(self.fc.servers, 'delete', side_effect=fakes_nova.fake_exception())

    @mock.patch.object(db_api, '_paginate_query')
    def test_filter_and_page_query_paginates_query(self, mock_paginate_query):
        query = mock.Mock()
        db_api._filter_and_page_query(self.ctx, query)
        self.assertTrue(mock_paginate_query.called)

    @mock.patch.object(db_api, '_events_paginate_query')
    def test_events_filter_and_page_query(self, mock_events_paginate_query):
        query = mock.Mock()
        db_api._events_filter_and_page_query(self.ctx, query)
        self.assertTrue(mock_events_paginate_query.called)

    @mock.patch.object(db_api.utils, 'paginate_query')
    def test_events_filter_invalid_sort_key(self, mock_paginate_query):
        query = mock.Mock()

        class InvalidSortKey(db_api.utils.InvalidSortKey):

            @property
            def message(_):
                self.fail("_events_paginate_query() should not have tried to access .message attribute - it's deprecated in oslo.db and removed from base Exception in Py3K.")
        mock_paginate_query.side_effect = InvalidSortKey()
        self.assertRaises(exception.Invalid, db_api._events_filter_and_page_query, self.ctx, query, sort_keys=['foo'])

    @mock.patch.object(db_api.db_filters, 'exact_filter')
    def test_filter_and_page_query_handles_no_filters(self, mock_db_filter):
        query = mock.Mock()
        db_api._filter_and_page_query(self.ctx, query)
        mock_db_filter.assert_called_once_with(mock.ANY, mock.ANY, {})

    @mock.patch.object(db_api.db_filters, 'exact_filter')
    def test_events_filter_and_page_query_handles_no_filters(self, mock_db_filter):
        query = mock.Mock()
        db_api._events_filter_and_page_query(self.ctx, query)
        mock_db_filter.assert_called_once_with(mock.ANY, mock.ANY, {})

    @mock.patch.object(db_api.db_filters, 'exact_filter')
    def test_filter_and_page_query_applies_filters(self, mock_db_filter):
        query = mock.Mock()
        filters = {'foo': 'bar'}
        db_api._filter_and_page_query(self.ctx, query, filters=filters)
        self.assertTrue(mock_db_filter.called)

    @mock.patch.object(db_api.db_filters, 'exact_filter')
    def test_events_filter_and_page_query_applies_filters(self, mock_db_filter):
        query = mock.Mock()
        filters = {'foo': 'bar'}
        db_api._events_filter_and_page_query(self.ctx, query, filters=filters)
        self.assertTrue(mock_db_filter.called)

    @mock.patch.object(db_api, '_paginate_query')
    def test_filter_and_page_query_allowed_sort_keys(self, mock_paginate_query):
        query = mock.Mock()
        sort_keys = ['stack_name', 'foo']
        db_api._filter_and_page_query(self.ctx, query, sort_keys=sort_keys)
        args, _ = mock_paginate_query.call_args
        self.assertIn(['name'], args)

    @mock.patch.object(db_api, '_events_paginate_query')
    def test_events_filter_and_page_query_allowed_sort_keys(self, mock_paginate_query):
        query = mock.Mock()
        sort_keys = ['event_time', 'foo']
        db_api._events_filter_and_page_query(self.ctx, query, sort_keys=sort_keys)
        args, _ = mock_paginate_query.call_args
        self.assertIn(['created_at'], args)

    @mock.patch.object(db_api.utils, 'paginate_query')
    def test_paginate_query_default_sorts_by_created_at_and_id(self, mock_paginate_query):
        query = mock.Mock()
        model = mock.Mock()
        db_api._paginate_query(self.ctx, query, model, sort_keys=None)
        args, _ = mock_paginate_query.call_args
        self.assertIn(['created_at', 'id'], args)

    @mock.patch.object(db_api.utils, 'paginate_query')
    def test_paginate_query_default_sorts_dir_by_desc(self, mock_paginate_query):
        query = mock.Mock()
        model = mock.Mock()
        db_api._paginate_query(self.ctx, query, model, sort_dir=None)
        args, _ = mock_paginate_query.call_args
        self.assertIn('desc', args)

    @mock.patch.object(db_api.utils, 'paginate_query')
    def test_paginate_query_uses_given_sort_plus_id(self, mock_paginate_query):
        query = mock.Mock()
        model = mock.Mock()
        db_api._paginate_query(self.ctx, query, model, sort_keys=['name'])
        args, _ = mock_paginate_query.call_args
        self.assertIn(['name', 'id'], args)

    @mock.patch.object(db_api.utils, 'paginate_query')
    def test_paginate_query_gets_model_marker(self, mock_paginate_query):
        query = mock.Mock()
        model = mock.Mock()
        marker = mock.Mock()
        result = 'real_marker'
        ctx = mock.MagicMock()
        ctx.session.get.return_value = result
        db_api._paginate_query(ctx, query, model, marker=marker)
        ctx.session.get.assert_called_once_with(model, marker)
        args, _ = mock_paginate_query.call_args
        self.assertIn(result, args)

    @mock.patch.object(db_api.utils, 'paginate_query')
    def test_paginate_query_raises_invalid_sort_key(self, mock_paginate_query):
        query = mock.Mock()
        model = mock.Mock()

        class InvalidSortKey(db_api.utils.InvalidSortKey):

            @property
            def message(_):
                self.fail("_paginate_query() should not have tried to access .message attribute - it's deprecated in oslo.db and removed from base Exception class in Py3K.")
        mock_paginate_query.side_effect = InvalidSortKey()
        self.assertRaises(exception.Invalid, db_api._paginate_query, self.ctx, query, model, sort_keys=['foo'])

    def test_get_sort_keys_returns_empty_list_if_no_keys(self):
        sort_keys = None
        mapping = {}
        filtered_keys = db_api._get_sort_keys(sort_keys, mapping)
        self.assertEqual([], filtered_keys)

    def test_get_sort_keys_allow_single_key(self):
        sort_key = 'foo'
        mapping = {'foo': 'Foo'}
        filtered_keys = db_api._get_sort_keys(sort_key, mapping)
        self.assertEqual(['Foo'], filtered_keys)

    def test_get_sort_keys_allow_multiple_keys(self):
        sort_keys = ['foo', 'bar', 'nope']
        mapping = {'foo': 'Foo', 'bar': 'Bar'}
        filtered_keys = db_api._get_sort_keys(sort_keys, mapping)
        self.assertIn('Foo', filtered_keys)
        self.assertIn('Bar', filtered_keys)
        self.assertEqual(2, len(filtered_keys))

    def test_encryption(self):
        stack_name = 'test_encryption'
        stack = self._setup_test_stack(stack_name)[1]
        self._mock_create()
        stack.create()
        stack = parser.Stack.load(self.ctx, stack.id)
        cs = stack['WebServer']
        cs.data_set('my_secret', 'fake secret', True)
        rs = db_api.resource_get_by_name_and_stack(self.ctx, 'WebServer', stack.id)
        encrypted_key = rs.data[0]['value']
        self.assertNotEqual(encrypted_key, 'fake secret')
        self.assertEqual('fake secret', db_api.resource_data_get(self.ctx, cs.id, 'my_secret'))
        self.assertEqual('fake secret', db_api.resource_data_get(self.ctx, cs.id, 'my_secret'))
        self.fc.servers.create.assert_called_once_with(image=744, flavor=3, key_name='test', name=mock.ANY, security_groups=None, userdata=mock.ANY, scheduler_hints=None, meta=None, nics=None, availability_zone=None, block_device_mapping=None)

    def test_resource_data_delete(self):
        stack = self._setup_test_stack('res_data_delete', UUID1)[1]
        self._mock_create()
        stack.create()
        stack = parser.Stack.load(self.ctx, stack.id)
        resource = stack['WebServer']
        resource.data_set('test', 'test_data')
        self.assertEqual('test_data', db_api.resource_data_get(self.ctx, resource.id, 'test'))
        db_api.resource_data_delete(self.ctx, resource.id, 'test')
        self.assertRaises(exception.NotFound, db_api.resource_data_get, self.ctx, resource.id, 'test')
        self.fc.servers.create.assert_called_once_with(image=744, flavor=3, key_name='test', name=mock.ANY, security_groups=None, userdata=mock.ANY, scheduler_hints=None, meta=None, nics=None, availability_zone=None, block_device_mapping=None)

    def test_stack_get_by_name(self):
        name = 'stack_get_by_name'
        stack = self._setup_test_stack(name, UUID1, stack_user_project_id=UUID2)[1]
        st = db_api.stack_get_by_name(self.ctx, name)
        self.assertEqual(UUID1, st.id)
        self.ctx.project_id = UUID3
        st = db_api.stack_get_by_name(self.ctx, name)
        self.assertIsNone(st)
        self.ctx.project_id = UUID2
        st = db_api.stack_get_by_name(self.ctx, name)
        self.assertEqual(UUID1, st.id)
        stack.delete()
        st = db_api.stack_get_by_name(self.ctx, name)
        self.assertIsNone(st)

    def test_stack_create_multiple(self):
        name = 'stack_race'
        stack = self._setup_test_stack(name, UUID1, stack_user_project_id=UUID2)[1]
        self.assertRaises(exception.StackExists, self._setup_test_stack, name, UUID2, stack_user_project_id=UUID2)
        st = db_api.stack_get_by_name(self.ctx, name)
        self.assertEqual(UUID1, st.id)
        stack.delete()
        self.assertIsNone(db_api.stack_get_by_name(self.ctx, name))

    def test_nested_stack_get_by_name(self):
        stack1 = self._setup_test_stack('neststack1', UUID1)[1]
        stack2 = self._setup_test_stack('neststack2', UUID2, owner_id=stack1.id)[1]
        result = db_api.stack_get_by_name(self.ctx, 'neststack2')
        self.assertEqual(UUID2, result.id)
        stack2.delete()
        result = db_api.stack_get_by_name(self.ctx, 'neststack2')
        self.assertIsNone(result)

    def test_stack_get_by_name_and_owner_id(self):
        stack1 = self._setup_test_stack('ownstack1', UUID1, stack_user_project_id=UUID3)[1]
        stack2 = self._setup_test_stack('ownstack2', UUID2, owner_id=stack1.id, stack_user_project_id=UUID3)[1]
        result = db_api.stack_get_by_name_and_owner_id(self.ctx, 'ownstack2', None)
        self.assertIsNone(result)
        result = db_api.stack_get_by_name_and_owner_id(self.ctx, 'ownstack2', stack1.id)
        self.assertEqual(UUID2, result.id)
        self.ctx.project_id = str(uuid.uuid4())
        result = db_api.stack_get_by_name_and_owner_id(self.ctx, 'ownstack2', None)
        self.assertIsNone(result)
        self.ctx.project_id = UUID3
        result = db_api.stack_get_by_name_and_owner_id(self.ctx, 'ownstack2', stack1.id)
        self.assertEqual(UUID2, result.id)
        stack2.delete()
        result = db_api.stack_get_by_name_and_owner_id(self.ctx, 'ownstack2', stack1.id)
        self.assertIsNone(result)

    def test_stack_get(self):
        stack = self._setup_test_stack('stack_get', UUID1)[1]
        st = db_api.stack_get(self.ctx, UUID1, show_deleted=False)
        self.assertEqual(UUID1, st.id)
        stack.delete()
        st = db_api.stack_get(self.ctx, UUID1, show_deleted=False)
        self.assertIsNone(st)
        st = db_api.stack_get(self.ctx, UUID1, show_deleted=True)
        self.assertEqual(UUID1, st.id)

    def test_stack_get_status(self):
        stack = self._setup_test_stack('stack_get_status', UUID1)[1]
        st = db_api.stack_get_status(self.ctx, UUID1)
        self.assertEqual(('CREATE', 'IN_PROGRESS', '', None), st)
        stack.delete()
        st = db_api.stack_get_status(self.ctx, UUID1)
        self.assertEqual(('DELETE', 'COMPLETE', 'Stack DELETE completed successfully', None), st)
        self.assertRaises(exception.NotFound, db_api.stack_get_status, self.ctx, UUID2)

    def test_stack_get_show_deleted_context(self):
        stack = self._setup_test_stack('stack_get_deleted', UUID1)[1]
        self.assertFalse(self.ctx.show_deleted)
        st = db_api.stack_get(self.ctx, UUID1)
        self.assertEqual(UUID1, st.id)
        stack.delete()
        st = db_api.stack_get(self.ctx, UUID1)
        self.assertIsNone(st)
        self.ctx.show_deleted = True
        st = db_api.stack_get(self.ctx, UUID1)
        self.assertEqual(UUID1, st.id)

    def test_stack_get_all(self):
        stacks = [self._setup_test_stack('stack_get_all_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        st_db = db_api.stack_get_all(self.ctx)
        self.assertEqual(3, len(st_db))
        stacks[0].delete()
        st_db = db_api.stack_get_all(self.ctx)
        self.assertEqual(2, len(st_db))
        stacks[1].delete()
        st_db = db_api.stack_get_all(self.ctx)
        self.assertEqual(1, len(st_db))

    def test_stack_get_all_show_deleted(self):
        stacks = [self._setup_test_stack('stack_get_all_deleted_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        st_db = db_api.stack_get_all(self.ctx)
        self.assertEqual(3, len(st_db))
        stacks[0].delete()
        st_db = db_api.stack_get_all(self.ctx)
        self.assertEqual(2, len(st_db))
        st_db = db_api.stack_get_all(self.ctx, show_deleted=True)
        self.assertEqual(3, len(st_db))

    def test_stack_get_all_show_nested(self):
        stack1 = self._setup_test_stack('neststack_get_all_1', UUID1)[1]
        stack2 = self._setup_test_stack('neststack_get_all_2', UUID2, owner_id=stack1.id)[1]
        stack3 = self._setup_test_stack('neststack_get_all_1*', UUID3, owner_id=stack1.id, backup=True)[1]
        st_db = db_api.stack_get_all(self.ctx)
        self.assertEqual(1, len(st_db))
        self.assertEqual(stack1.id, st_db[0].id)
        st_db = db_api.stack_get_all(self.ctx, show_nested=True)
        self.assertEqual(2, len(st_db))
        st_ids = [s.id for s in st_db]
        self.assertNotIn(stack3.id, st_ids)
        self.assertIn(stack1.id, st_ids)
        self.assertIn(stack2.id, st_ids)

    def test_stack_get_all_with_filters(self):
        self._setup_test_stack('foo', UUID1)
        self._setup_test_stack('baz', UUID2)
        filters = {'name': 'foo'}
        results = db_api.stack_get_all(self.ctx, filters=filters)
        self.assertEqual(1, len(results))
        self.assertEqual('foo', results[0]['name'])

    def test_stack_get_all_filter_matches_in_list(self):
        self._setup_test_stack('wibble', UUID1)
        self._setup_test_stack('bar', UUID2)
        filters = {'name': ['bar', 'quux']}
        results = db_api.stack_get_all(self.ctx, filters=filters)
        self.assertEqual(1, len(results))
        self.assertEqual('bar', results[0]['name'])

    def test_stack_get_all_returns_all_if_no_filters(self):
        self._setup_test_stack('stack_get_all_no_filter1', UUID1)
        self._setup_test_stack('stack_get_all_no_filter2', UUID2)
        filters = None
        results = db_api.stack_get_all(self.ctx, filters=filters)
        self.assertEqual(2, len(results))

    def test_stack_get_all_default_sort_keys_and_dir(self):
        stacks = [self._setup_test_stack('stacks_def_sort_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        st_db = db_api.stack_get_all(self.ctx)
        self.assertEqual(3, len(st_db))
        self.assertEqual(stacks[2].id, st_db[0].id)
        self.assertEqual(stacks[1].id, st_db[1].id)
        self.assertEqual(stacks[0].id, st_db[2].id)

    def test_stack_get_all_default_sort_dir(self):
        stacks = [self._setup_test_stack('stacks_def_sort_dir_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        st_db = db_api.stack_get_all(self.ctx, sort_dir='asc')
        self.assertEqual(3, len(st_db))
        self.assertEqual(stacks[0].id, st_db[0].id)
        self.assertEqual(stacks[1].id, st_db[1].id)
        self.assertEqual(stacks[2].id, st_db[2].id)

    def test_stack_get_all_str_sort_keys(self):
        stacks = [self._setup_test_stack('stacks_str_sort_keys_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        st_db = db_api.stack_get_all(self.ctx, sort_keys='creation_time')
        self.assertEqual(3, len(st_db))
        self.assertEqual(stacks[0].id, st_db[0].id)
        self.assertEqual(stacks[1].id, st_db[1].id)
        self.assertEqual(stacks[2].id, st_db[2].id)

    @mock.patch.object(db_api.utils, 'paginate_query')
    def test_stack_get_all_filters_sort_keys(self, mock_paginate):
        sort_keys = ['stack_name', 'stack_status', 'creation_time', 'updated_time', 'stack_owner']
        db_api.stack_get_all(self.ctx, sort_keys=sort_keys)
        args = mock_paginate.call_args[0]
        used_sort_keys = set(args[3])
        expected_keys = set(['name', 'status', 'created_at', 'updated_at', 'id'])
        self.assertEqual(expected_keys, used_sort_keys)

    def test_stack_get_all_marker(self):
        stacks = [self._setup_test_stack('stacks_marker_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        st_db = db_api.stack_get_all(self.ctx, marker=stacks[1].id)
        self.assertEqual(1, len(st_db))
        self.assertEqual(stacks[0].id, st_db[0].id)

    def test_stack_get_all_non_existing_marker(self):
        [self._setup_test_stack('stacks_nonex_marker_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        uuid = "this stack doesn't exist"
        st_db = db_api.stack_get_all(self.ctx, marker=uuid)
        self.assertEqual(3, len(st_db))

    def test_stack_get_all_doesnt_mutate_sort_keys(self):
        [self._setup_test_stack('stacks_sort_nomutate_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        sort_keys = ['id']
        db_api.stack_get_all(self.ctx, sort_keys=sort_keys)
        self.assertEqual(['id'], sort_keys)

    def test_stack_get_all_hidden_tags(self):
        cfg.CONF.set_override('hidden_stack_tags', ['hidden'])
        stacks = [self._setup_test_stack('stacks_hidden_tags_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        stacks[0].tags = ['hidden']
        stacks[0].store()
        stacks[1].tags = ['random']
        stacks[1].store()
        st_db = db_api.stack_get_all(self.ctx, show_hidden=True)
        self.assertEqual(3, len(st_db))
        st_db_visible = db_api.stack_get_all(self.ctx, show_hidden=False)
        self.assertEqual(2, len(st_db_visible))
        for stack in st_db_visible:
            self.assertNotEqual(stacks[0].id, stack.id)

    def test_stack_get_all_by_tags(self):
        stacks = [self._setup_test_stack('stacks_tags_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        stacks[0].tags = ['tag1']
        stacks[0].store()
        stacks[1].tags = ['tag1', 'tag2']
        stacks[1].store()
        stacks[2].tags = ['tag1', 'tag2', 'tag3']
        stacks[2].store()
        st_db = db_api.stack_get_all(self.ctx, tags=['tag2'])
        self.assertEqual(2, len(st_db))
        st_db = db_api.stack_get_all(self.ctx, tags=['tag1', 'tag2'])
        self.assertEqual(2, len(st_db))
        st_db = db_api.stack_get_all(self.ctx, tags=['tag1', 'tag2', 'tag3'])
        self.assertEqual(1, len(st_db))

    def test_stack_get_all_by_tags_any(self):
        stacks = [self._setup_test_stack('stacks_tags_any_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        stacks[0].tags = ['tag2']
        stacks[0].store()
        stacks[1].tags = ['tag1', 'tag2']
        stacks[1].store()
        stacks[2].tags = ['tag1', 'tag3']
        stacks[2].store()
        st_db = db_api.stack_get_all(self.ctx, tags_any=['tag1'])
        self.assertEqual(2, len(st_db))
        st_db = db_api.stack_get_all(self.ctx, tags_any=['tag1', 'tag2', 'tag3'])
        self.assertEqual(3, len(st_db))

    def test_stack_get_all_by_not_tags(self):
        stacks = [self._setup_test_stack('stacks_not_tags_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        stacks[0].tags = ['tag1']
        stacks[0].store()
        stacks[1].tags = ['tag1', 'tag2']
        stacks[1].store()
        stacks[2].tags = ['tag1', 'tag2', 'tag3']
        stacks[2].store()
        st_db = db_api.stack_get_all(self.ctx, not_tags=['tag2'])
        self.assertEqual(1, len(st_db))
        st_db = db_api.stack_get_all(self.ctx, not_tags=['tag1', 'tag2'])
        self.assertEqual(1, len(st_db))
        st_db = db_api.stack_get_all(self.ctx, not_tags=['tag1', 'tag2', 'tag3'])
        self.assertEqual(2, len(st_db))

    def test_stack_get_all_by_not_tags_any(self):
        stacks = [self._setup_test_stack('stacks_not_tags_any_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        stacks[0].tags = ['tag2']
        stacks[0].store()
        stacks[1].tags = ['tag1', 'tag2']
        stacks[1].store()
        stacks[2].tags = ['tag1', 'tag3']
        stacks[2].store()
        st_db = db_api.stack_get_all(self.ctx, not_tags_any=['tag1'])
        self.assertEqual(1, len(st_db))
        st_db = db_api.stack_get_all(self.ctx, not_tags_any=['tag1', 'tag2', 'tag3'])
        self.assertEqual(0, len(st_db))

    def test_stack_get_all_by_tag_with_pagination(self):
        stacks = [self._setup_test_stack('stacks_tag_page_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        stacks[0].tags = ['tag1']
        stacks[0].store()
        stacks[1].tags = ['tag2']
        stacks[1].store()
        stacks[2].tags = ['tag1']
        stacks[2].store()
        st_db = db_api.stack_get_all(self.ctx, tags=['tag1'])
        self.assertEqual(2, len(st_db))
        st_db = db_api.stack_get_all(self.ctx, tags=['tag1'], limit=1)
        self.assertEqual(1, len(st_db))
        self.assertEqual(stacks[2].id, st_db[0].id)
        st_db = db_api.stack_get_all(self.ctx, tags=['tag1'], limit=1, marker=stacks[2].id)
        self.assertEqual(1, len(st_db))
        self.assertEqual(stacks[0].id, st_db[0].id)

    def test_stack_get_all_by_tag_with_show_hidden(self):
        cfg.CONF.set_override('hidden_stack_tags', ['hidden'])
        stacks = [self._setup_test_stack('stacks_tag_hidden_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        stacks[0].tags = ['tag1']
        stacks[0].store()
        stacks[1].tags = ['hidden', 'tag1']
        stacks[1].store()
        st_db = db_api.stack_get_all(self.ctx, tags=['tag1'], show_hidden=True)
        self.assertEqual(2, len(st_db))
        st_db = db_api.stack_get_all(self.ctx, tags=['tag1'], show_hidden=False)
        self.assertEqual(1, len(st_db))

    def test_stack_count_all(self):
        stacks = [self._setup_test_stack('stacks_count_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        st_db = db_api.stack_count_all(self.ctx)
        self.assertEqual(3, st_db)
        stacks[0].delete()
        st_db = db_api.stack_count_all(self.ctx)
        self.assertEqual(2, st_db)
        st_db = db_api.stack_count_all(self.ctx, show_deleted=True)
        self.assertEqual(3, st_db)
        stacks[1].delete()
        st_db = db_api.stack_count_all(self.ctx)
        self.assertEqual(1, st_db)
        st_db = db_api.stack_count_all(self.ctx, show_deleted=True)
        self.assertEqual(3, st_db)

    def test_count_all_hidden_tags(self):
        cfg.CONF.set_override('hidden_stack_tags', ['hidden'])
        stacks = [self._setup_test_stack('stacks_count_hid_tag_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        stacks[0].tags = ['hidden']
        stacks[0].store()
        stacks[1].tags = ['random']
        stacks[1].store()
        st_db = db_api.stack_count_all(self.ctx, show_hidden=True)
        self.assertEqual(3, st_db)
        st_db_visible = db_api.stack_count_all(self.ctx, show_hidden=False)
        self.assertEqual(2, st_db_visible)

    def test_count_all_by_tags(self):
        stacks = [self._setup_test_stack('stacks_count_all_tag_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        stacks[0].tags = ['tag1']
        stacks[0].store()
        stacks[1].tags = ['tag2']
        stacks[1].store()
        stacks[2].tags = ['tag2']
        stacks[2].store()
        st_db = db_api.stack_count_all(self.ctx, tags=['tag1'])
        self.assertEqual(1, st_db)
        st_db = db_api.stack_count_all(self.ctx, tags=['tag2'])
        self.assertEqual(2, st_db)

    def test_count_all_by_tag_with_show_hidden(self):
        cfg.CONF.set_override('hidden_stack_tags', ['hidden'])
        stacks = [self._setup_test_stack('stacks_count_all_tagsh_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        stacks[0].tags = ['tag1']
        stacks[0].store()
        stacks[1].tags = ['hidden', 'tag1']
        stacks[1].store()
        st_db = db_api.stack_count_all(self.ctx, tags=['tag1'], show_hidden=True)
        self.assertEqual(2, st_db)
        st_db = db_api.stack_count_all(self.ctx, tags=['tag1'], show_hidden=False)
        self.assertEqual(1, st_db)

    def test_stack_count_all_with_filters(self):
        self._setup_test_stack('sca_foo', UUID1)
        self._setup_test_stack('sca_bar', UUID2)
        filters = {'name': 'sca_bar'}
        st_db = db_api.stack_count_all(self.ctx, filters=filters)
        self.assertEqual(1, st_db)

    def test_stack_count_all_show_nested(self):
        stack1 = self._setup_test_stack('stack1', UUID1)[1]
        self._setup_test_stack('stack2', UUID2, owner_id=stack1.id)
        self._setup_test_stack('stack1*', UUID3, owner_id=stack1.id, backup=True)
        st_db = db_api.stack_count_all(self.ctx)
        self.assertEqual(1, st_db)
        st_db = db_api.stack_count_all(self.ctx, show_nested=True)
        self.assertEqual(2, st_db)

    def test_event_get_all_by_stack(self):
        stack = self._setup_test_stack('stack_events', UUID1)[1]
        self._mock_create()
        stack.create()
        stack._persist_state()
        events = db_api.event_get_all_by_stack(self.ctx, UUID1)
        self.assertEqual(4, len(events))
        filters = {'resource_status': 'COMPLETE'}
        events = db_api.event_get_all_by_stack(self.ctx, UUID1, filters=filters)
        self.assertEqual(2, len(events))
        self.assertEqual('COMPLETE', events[0].resource_status)
        self.assertEqual('COMPLETE', events[1].resource_status)
        filters = {'resource_action': 'CREATE'}
        events = db_api.event_get_all_by_stack(self.ctx, UUID1, filters=filters)
        self.assertEqual(4, len(events))
        self.assertEqual('CREATE', events[0].resource_action)
        self.assertEqual('CREATE', events[1].resource_action)
        self.assertEqual('CREATE', events[2].resource_action)
        self.assertEqual('CREATE', events[3].resource_action)
        filters = {'resource_type': 'AWS::EC2::Instance'}
        events = db_api.event_get_all_by_stack(self.ctx, UUID1, filters=filters)
        self.assertEqual(2, len(events))
        self.assertEqual('AWS::EC2::Instance', events[0].resource_type)
        self.assertEqual('AWS::EC2::Instance', events[1].resource_type)
        filters = {'resource_type': 'OS::Nova::Server'}
        events = db_api.event_get_all_by_stack(self.ctx, UUID1, filters=filters)
        self.assertEqual(0, len(events))
        events_all = db_api.event_get_all_by_stack(self.ctx, UUID1)
        marker = events_all[0].uuid
        expected = events_all[1].uuid
        events = db_api.event_get_all_by_stack(self.ctx, UUID1, limit=1, marker=marker)
        self.assertEqual(1, len(events))
        self.assertEqual(expected, events[0].uuid)
        self._mock_delete()
        stack.delete()
        filters = {'resource_status': 'COMPLETE'}
        events = db_api.event_get_all_by_stack(self.ctx, UUID1, filters=filters)
        self.assertEqual(4, len(events))
        self.assertEqual('COMPLETE', events[0].resource_status)
        self.assertEqual('COMPLETE', events[1].resource_status)
        self.assertEqual('COMPLETE', events[2].resource_status)
        self.assertEqual('COMPLETE', events[3].resource_status)
        filters = {'resource_action': 'DELETE', 'resource_status': 'COMPLETE'}
        events = db_api.event_get_all_by_stack(self.ctx, UUID1, filters=filters)
        self.assertEqual(2, len(events))
        self.assertEqual('DELETE', events[0].resource_action)
        self.assertEqual('COMPLETE', events[0].resource_status)
        self.assertEqual('DELETE', events[1].resource_action)
        self.assertEqual('COMPLETE', events[1].resource_status)
        events_all = db_api.event_get_all_by_stack(self.ctx, UUID1)
        self.assertEqual(8, len(events_all))
        marker = events_all[1].uuid
        events2_uuid = events_all[2].uuid
        events3_uuid = events_all[3].uuid
        events = db_api.event_get_all_by_stack(self.ctx, UUID1, limit=1, marker=marker)
        self.assertEqual(1, len(events))
        self.assertEqual(events2_uuid, events[0].uuid)
        events = db_api.event_get_all_by_stack(self.ctx, UUID1, limit=2, marker=marker)
        self.assertEqual(2, len(events))
        self.assertEqual(events2_uuid, events[0].uuid)
        self.assertEqual(events3_uuid, events[1].uuid)
        self.fc.servers.create.assert_called_once_with(image=744, flavor=3, key_name='test', name=mock.ANY, security_groups=None, userdata=mock.ANY, scheduler_hints=None, meta=None, nics=None, availability_zone=None, block_device_mapping=None)

    def test_event_count_all_by_stack(self):
        stack = self._setup_test_stack('stack_event_count', UUID1)[1]
        self._mock_create()
        stack.create()
        stack._persist_state()
        num_events = db_api.event_count_all_by_stack(self.ctx, UUID1)
        self.assertEqual(4, num_events)
        self._mock_delete()
        stack.delete()
        num_events = db_api.event_count_all_by_stack(self.ctx, UUID1)
        self.assertEqual(8, num_events)
        self.fc.servers.create.assert_called_once_with(image=744, flavor=3, key_name='test', name=mock.ANY, security_groups=None, userdata=mock.ANY, scheduler_hints=None, meta=None, nics=None, availability_zone=None, block_device_mapping=None)

    def test_event_get_all_by_tenant(self):
        stacks = [self._setup_test_stack('stack_ev_ten_%d' % i, x)[1] for i, x in enumerate(UUIDs)]
        self._mock_create()
        [s.create() for s in stacks]
        [s._persist_state() for s in stacks]
        events = db_api.event_get_all_by_tenant(self.ctx)
        self.assertEqual(12, len(events))
        self._mock_delete()
        [s.delete() for s in stacks]
        events = db_api.event_get_all_by_tenant(self.ctx)
        self.assertEqual(0, len(events))
        self.fc.servers.create.assert_called_with(image=744, flavor=3, key_name='test', name=mock.ANY, security_groups=None, userdata=mock.ANY, scheduler_hints=None, meta=None, nics=None, availability_zone=None, block_device_mapping=None)
        self.assertEqual(len(stacks), self.fc.servers.create.call_count)

    def test_user_creds_password(self):
        self.ctx.password = 'password'
        self.ctx.trust_id = None
        self.ctx.region_name = 'RegionOne'
        db_creds = db_api.user_creds_create(self.ctx)
        load_creds = db_api.user_creds_get(self.ctx, db_creds['id'])
        self.assertEqual('test_username', load_creds.get('username'))
        self.assertEqual('password', load_creds.get('password'))
        self.assertEqual('test_tenant', load_creds.get('tenant'))
        self.assertEqual('test_tenant_id', load_creds.get('tenant_id'))
        self.assertEqual('RegionOne', load_creds.get('region_name'))
        self.assertIsNotNone(load_creds.get('created_at'))
        self.assertIsNone(load_creds.get('updated_at'))
        self.assertEqual('http://server.test:5000/v2.0', load_creds.get('auth_url'))
        self.assertIsNone(load_creds.get('trust_id'))
        self.assertIsNone(load_creds.get('trustor_user_id'))

    def test_user_creds_password_too_long(self):
        self.ctx.trust_id = None
        self.ctx.password = 'O123456789O1234567' * 20
        error = self.assertRaises(exception.Error, db_api.user_creds_create, self.ctx)
        self.assertIn('Length of OS_PASSWORD after encryption exceeds Heat limit (255 chars)', str(error))

    def test_user_creds_trust(self):
        self.ctx.username = None
        self.ctx.password = None
        self.ctx.trust_id = 'atrust123'
        self.ctx.trustor_user_id = 'atrustor123'
        self.ctx.project_id = 'atenant123'
        self.ctx.project_name = 'atenant'
        self.ctx.auth_url = 'anauthurl'
        self.ctx.region_name = 'aregion'
        db_creds = db_api.user_creds_create(self.ctx)
        load_creds = db_api.user_creds_get(self.ctx, db_creds['id'])
        self.assertIsNone(load_creds.get('username'))
        self.assertIsNone(load_creds.get('password'))
        self.assertIsNotNone(load_creds.get('created_at'))
        self.assertIsNone(load_creds.get('updated_at'))
        self.assertEqual('anauthurl', load_creds.get('auth_url'))
        self.assertEqual('aregion', load_creds.get('region_name'))
        self.assertEqual('atenant123', load_creds.get('tenant_id'))
        self.assertEqual('atenant', load_creds.get('tenant'))
        self.assertEqual('atrust123', load_creds.get('trust_id'))
        self.assertEqual('atrustor123', load_creds.get('trustor_user_id'))

    def test_user_creds_none(self):
        self.ctx.username = None
        self.ctx.password = None
        self.ctx.trust_id = None
        self.ctx.region_name = None
        db_creds = db_api.user_creds_create(self.ctx)
        load_creds = db_api.user_creds_get(self.ctx, db_creds['id'])
        self.assertIsNone(load_creds.get('username'))
        self.assertIsNone(load_creds.get('password'))
        self.assertIsNone(load_creds.get('trust_id'))
        self.assertIsNone(load_creds.get('region_name'))

    def test_software_config_create(self):
        tenant_id = self.ctx.tenant_id
        config = db_api.software_config_create(self.ctx, {'name': 'config_mysql', 'tenant': tenant_id})
        self.assertIsNotNone(config)
        self.assertEqual('config_mysql', config.name)
        self.assertEqual(tenant_id, config.tenant)

    def test_software_config_get(self):
        self.assertRaises(exception.NotFound, db_api.software_config_get, self.ctx, str(uuid.uuid4()))
        conf = '#!/bin/bash\necho "$bar and $foo"\n'
        config = {'inputs': [{'name': 'foo'}, {'name': 'bar'}], 'outputs': [{'name': 'result'}], 'config': conf, 'options': {}}
        tenant_id = self.ctx.tenant_id
        values = {'name': 'config_mysql', 'tenant': tenant_id, 'group': 'Heat::Shell', 'config': config}
        config = db_api.software_config_create(self.ctx, values)
        config_id = config.id
        config = db_api.software_config_get(self.ctx, config_id)
        self.assertIsNotNone(config)
        self.assertEqual('config_mysql', config.name)
        self.assertEqual(tenant_id, config.tenant)
        self.assertEqual('Heat::Shell', config.group)
        self.assertEqual(conf, config.config['config'])
        self.ctx.project_id = None
        self.assertRaises(exception.NotFound, db_api.software_config_get, self.ctx, config_id)
        admin_ctx = utils.dummy_context(is_admin=True, tenant_id='admin_tenant')
        config = db_api.software_config_get(admin_ctx, config_id)
        self.assertIsNotNone(config)

    def _create_software_config_record(self):
        tenant_id = self.ctx.tenant_id
        software_config = db_api.software_config_create(self.ctx, {'name': 'config_mysql', 'tenant': tenant_id})
        self.assertIsNotNone(software_config)
        return software_config.id

    def _test_software_config_get_all(self, get_ctx=None):
        self.assertEqual([], db_api.software_config_get_all(self.ctx))
        scf_id = self._create_software_config_record()
        software_configs = db_api.software_config_get_all(get_ctx or self.ctx)
        self.assertEqual(1, len(software_configs))
        self.assertEqual(scf_id, software_configs[0].id)

    def test_software_config_get_all(self):
        self._test_software_config_get_all()

    def test_software_config_get_all_by_admin(self):
        admin_ctx = utils.dummy_context(is_admin=True, tenant_id='admin_tenant')
        self._test_software_config_get_all(get_ctx=admin_ctx)

    def test_software_config_count_all(self):
        self.assertEqual(0, db_api.software_config_count_all(self.ctx))
        self._create_software_config_record()
        self._create_software_config_record()
        self._create_software_config_record()
        self.assertEqual(3, db_api.software_config_count_all(self.ctx))

    def test_software_config_delete(self):
        scf_id = self._create_software_config_record()
        cfg = db_api.software_config_get(self.ctx, scf_id)
        self.assertIsNotNone(cfg)
        db_api.software_config_delete(self.ctx, scf_id)
        err = self.assertRaises(exception.NotFound, db_api.software_config_get, self.ctx, scf_id)
        self.assertIn(scf_id, str(err))
        err = self.assertRaises(exception.NotFound, db_api.software_config_delete, self.ctx, scf_id)
        self.assertIn(scf_id, str(err))

    def test_software_config_delete_by_admin(self):
        scf_id = self._create_software_config_record()
        cfg = db_api.software_config_get(self.ctx, scf_id)
        self.assertIsNotNone(cfg)
        admin_ctx = utils.dummy_context(is_admin=True, tenant_id='admin_tenant')
        db_api.software_config_delete(admin_ctx, scf_id)

    def test_software_config_delete_not_allowed(self):
        tenant_id = self.ctx.tenant_id
        config = db_api.software_config_create(self.ctx, {'name': 'config_mysql', 'tenant': tenant_id})
        config_id = config.id
        values = {'tenant': tenant_id, 'stack_user_project_id': str(uuid.uuid4()), 'config_id': config_id, 'server_id': str(uuid.uuid4())}
        db_api.software_deployment_create(self.ctx, values)
        err = self.assertRaises(exception.InvalidRestrictedAction, db_api.software_config_delete, self.ctx, config_id)
        msg = 'Software config with id %s can not be deleted as it is referenced' % config_id
        self.assertIn(msg, str(err))

    def _deployment_values(self):
        tenant_id = self.ctx.tenant_id
        stack_user_project_id = str(uuid.uuid4())
        config_id = db_api.software_config_create(self.ctx, {'name': 'config_mysql', 'tenant': tenant_id}).id
        server_id = str(uuid.uuid4())
        input_values = {'foo': 'fooooo', 'bar': 'baaaaa'}
        values = {'tenant': tenant_id, 'stack_user_project_id': stack_user_project_id, 'config_id': config_id, 'server_id': server_id, 'input_values': input_values}
        return values

    def test_software_deployment_create(self):
        values = self._deployment_values()
        deployment = db_api.software_deployment_create(self.ctx, values)
        self.assertIsNotNone(deployment)
        self.assertEqual(values['tenant'], deployment.tenant)

    def test_software_deployment_get(self):
        self.assertRaises(exception.NotFound, db_api.software_deployment_get, self.ctx, str(uuid.uuid4()))
        values = self._deployment_values()
        deployment = db_api.software_deployment_create(self.ctx, values)
        self.assertIsNotNone(deployment)
        deployment_id = deployment.id
        deployment = db_api.software_deployment_get(self.ctx, deployment_id)
        self.assertIsNotNone(deployment)
        self.assertEqual(values['tenant'], deployment.tenant)
        self.assertEqual(values['config_id'], deployment.config_id)
        self.assertEqual(values['server_id'], deployment.server_id)
        self.assertEqual(values['input_values'], deployment.input_values)
        self.assertEqual(values['stack_user_project_id'], deployment.stack_user_project_id)
        self.ctx.project_id = str(uuid.uuid4())
        self.assertRaises(exception.NotFound, db_api.software_deployment_get, self.ctx, deployment_id)
        self.ctx.project_id = deployment.stack_user_project_id
        deployment = db_api.software_deployment_get(self.ctx, deployment_id)
        self.assertIsNotNone(deployment)
        self.assertEqual(values['tenant'], deployment.tenant)
        admin_ctx = utils.dummy_context(is_admin=True, tenant_id='admin_tenant')
        deployment = db_api.software_deployment_get(admin_ctx, deployment_id)
        self.assertIsNotNone(deployment)

    def test_software_deployment_get_all(self):
        self.assertEqual([], db_api.software_deployment_get_all(self.ctx))
        values = self._deployment_values()
        deployment = db_api.software_deployment_create(self.ctx, values)
        self.assertIsNotNone(deployment)
        deployments = db_api.software_deployment_get_all(self.ctx)
        self.assertEqual(1, len(deployments))
        self.assertEqual(deployment.id, deployments[0].id)
        deployments = db_api.software_deployment_get_all(self.ctx, server_id=values['server_id'])
        self.assertEqual(1, len(deployments))
        self.assertEqual(deployment.id, deployments[0].id)
        deployments = db_api.software_deployment_get_all(self.ctx, server_id=str(uuid.uuid4()))
        self.assertEqual([], deployments)
        admin_ctx = utils.dummy_context(is_admin=True, tenant_id='admin_tenant')
        deployments = db_api.software_deployment_get_all(admin_ctx)
        self.assertEqual(1, len(deployments))

    def test_software_deployment_count_all(self):
        self.assertEqual(0, db_api.software_deployment_count_all(self.ctx))
        values = self._deployment_values()
        deployment = db_api.software_deployment_create(self.ctx, values)
        self.assertIsNotNone(deployment)
        deployment = db_api.software_deployment_create(self.ctx, values)
        self.assertIsNotNone(deployment)
        deployment = db_api.software_deployment_create(self.ctx, values)
        self.assertIsNotNone(deployment)
        self.assertEqual(3, db_api.software_deployment_count_all(self.ctx))

    def test_software_deployment_update(self):
        deployment_id = str(uuid.uuid4())
        err = self.assertRaises(exception.NotFound, db_api.software_deployment_update, self.ctx, deployment_id, values={})
        self.assertIn(deployment_id, str(err))
        values = self._deployment_values()
        deployment = db_api.software_deployment_create(self.ctx, values)
        deployment_id = deployment.id
        values = {'status': 'COMPLETED'}
        deployment = db_api.software_deployment_update(self.ctx, deployment_id, values)
        self.assertIsNotNone(deployment)
        self.assertEqual(values['status'], deployment.status)
        admin_ctx = utils.dummy_context(is_admin=True, tenant_id='admin_tenant')
        values = {'status': 'FAILED'}
        deployment = db_api.software_deployment_update(admin_ctx, deployment_id, values)
        self.assertIsNotNone(deployment)
        self.assertEqual(values['status'], deployment.status)

    def _test_software_deployment_delete(self, test_ctx=None):
        deployment_id = str(uuid.uuid4())
        err = self.assertRaises(exception.NotFound, db_api.software_deployment_delete, self.ctx, deployment_id)
        self.assertIn(deployment_id, str(err))
        values = self._deployment_values()
        deployment = db_api.software_deployment_create(self.ctx, values)
        deployment_id = deployment.id
        test_ctx = test_ctx or self.ctx
        deployment = db_api.software_deployment_get(test_ctx, deployment_id)
        self.assertIsNotNone(deployment)
        db_api.software_deployment_delete(test_ctx, deployment_id)
        err = self.assertRaises(exception.NotFound, db_api.software_deployment_get, test_ctx, deployment_id)
        self.assertIn(deployment_id, str(err))

    def test_software_deployment_delete(self):
        self._test_software_deployment_delete()

    def test_software_deployment_delete_by_admin(self):
        admin_ctx = utils.dummy_context(is_admin=True, tenant_id='admin_tenant')
        self._test_software_deployment_delete(test_ctx=admin_ctx)

    def test_snapshot_create(self):
        template = create_raw_template(self.ctx)
        user_creds = create_user_creds(self.ctx)
        stack = create_stack(self.ctx, template, user_creds)
        values = {'tenant': self.ctx.tenant_id, 'status': 'IN_PROGRESS', 'stack_id': stack.id}
        snapshot = db_api.snapshot_create(self.ctx, values)
        self.assertIsNotNone(snapshot)
        self.assertEqual(values['tenant'], snapshot.tenant)

    def test_snapshot_create_with_name(self):
        template = create_raw_template(self.ctx)
        user_creds = create_user_creds(self.ctx)
        stack = create_stack(self.ctx, template, user_creds)
        values = {'tenant': self.ctx.tenant_id, 'status': 'IN_PROGRESS', 'stack_id': stack.id, 'name': 'snap1'}
        snapshot = db_api.snapshot_create(self.ctx, values)
        self.assertIsNotNone(snapshot)
        self.assertEqual(values['tenant'], snapshot.tenant)
        self.assertEqual('snap1', snapshot.name)

    def test_snapshot_get_not_found(self):
        self.assertRaises(exception.NotFound, db_api.snapshot_get, self.ctx, str(uuid.uuid4()))

    def test_snapshot_get(self):
        template = create_raw_template(self.ctx)
        user_creds = create_user_creds(self.ctx)
        stack = create_stack(self.ctx, template, user_creds)
        values = {'tenant': self.ctx.tenant_id, 'status': 'IN_PROGRESS', 'stack_id': stack.id}
        snapshot = db_api.snapshot_create(self.ctx, values)
        self.assertIsNotNone(snapshot)
        snapshot_id = snapshot.id
        snapshot = db_api.snapshot_get(self.ctx, snapshot_id)
        self.assertIsNotNone(snapshot)
        self.assertEqual(values['tenant'], snapshot.tenant)
        self.assertEqual(values['status'], snapshot.status)
        self.assertIsNotNone(snapshot.created_at)

    def test_snapshot_get_by_another_stack(self):
        template = create_raw_template(self.ctx)
        user_creds = create_user_creds(self.ctx)
        stack = create_stack(self.ctx, template, user_creds)
        stack1 = create_stack(self.ctx, template, user_creds)
        values = {'tenant': self.ctx.tenant_id, 'status': 'IN_PROGRESS', 'stack_id': stack.id}
        snapshot = db_api.snapshot_create(self.ctx, values)
        self.assertIsNotNone(snapshot)
        snapshot_id = snapshot.id
        self.assertRaises(exception.SnapshotNotFound, db_api.snapshot_get_by_stack, self.ctx, snapshot_id, stack1)

    def test_snapshot_get_not_found_invalid_tenant(self):
        template = create_raw_template(self.ctx)
        user_creds = create_user_creds(self.ctx)
        stack = create_stack(self.ctx, template, user_creds)
        values = {'tenant': self.ctx.tenant_id, 'status': 'IN_PROGRESS', 'stack_id': stack.id}
        snapshot = db_api.snapshot_create(self.ctx, values)
        self.ctx.project_id = str(uuid.uuid4())
        self.assertRaises(exception.NotFound, db_api.snapshot_get, self.ctx, snapshot.id)

    def test_snapshot_update_not_found(self):
        snapshot_id = str(uuid.uuid4())
        err = self.assertRaises(exception.NotFound, db_api.snapshot_update, self.ctx, snapshot_id, values={})
        self.assertIn(snapshot_id, str(err))

    def test_snapshot_update(self):
        template = create_raw_template(self.ctx)
        user_creds = create_user_creds(self.ctx)
        stack = create_stack(self.ctx, template, user_creds)
        values = {'tenant': self.ctx.tenant_id, 'status': 'IN_PROGRESS', 'stack_id': stack.id}
        snapshot = db_api.snapshot_create(self.ctx, values)
        snapshot_id = snapshot.id
        values = {'status': 'COMPLETED'}
        snapshot = db_api.snapshot_update(self.ctx, snapshot_id, values)
        self.assertIsNotNone(snapshot)
        self.assertEqual(values['status'], snapshot.status)

    def test_snapshot_delete_not_found(self):
        snapshot_id = str(uuid.uuid4())
        err = self.assertRaises(exception.NotFound, db_api.snapshot_delete, self.ctx, snapshot_id)
        self.assertIn(snapshot_id, str(err))

    def test_snapshot_delete(self):
        template = create_raw_template(self.ctx)
        user_creds = create_user_creds(self.ctx)
        stack = create_stack(self.ctx, template, user_creds)
        values = {'tenant': self.ctx.tenant_id, 'status': 'IN_PROGRESS', 'stack_id': stack.id}
        snapshot = db_api.snapshot_create(self.ctx, values)
        snapshot_id = snapshot.id
        snapshot = db_api.snapshot_get(self.ctx, snapshot_id)
        self.assertIsNotNone(snapshot)
        db_api.snapshot_delete(self.ctx, snapshot_id)
        err = self.assertRaises(exception.NotFound, db_api.snapshot_get, self.ctx, snapshot_id)
        self.assertIn(snapshot_id, str(err))

    def test_snapshot_get_all_by_stack(self):
        template = create_raw_template(self.ctx)
        user_creds = create_user_creds(self.ctx)
        stack = create_stack(self.ctx, template, user_creds)
        values = {'tenant': self.ctx.tenant_id, 'status': 'IN_PROGRESS', 'stack_id': stack.id}
        snapshot = db_api.snapshot_create(self.ctx, values)
        self.assertIsNotNone(snapshot)
        [snapshot] = db_api.snapshot_get_all_by_stack(self.ctx, stack.id)
        self.assertIsNotNone(snapshot)
        self.assertEqual(values['tenant'], snapshot.tenant)
        self.assertEqual(values['status'], snapshot.status)
        self.assertIsNotNone(snapshot.created_at)

    def test_snapshot_count_all_by_stack(self):
        template = create_raw_template(self.ctx)
        user_creds = create_user_creds(self.ctx)
        stack1 = create_stack(self.ctx, template, user_creds)
        stack2 = create_stack(self.ctx, template, user_creds)
        values = [{'tenant': self.ctx.tenant_id, 'status': 'IN_PROGRESS', 'stack_id': stack1.id, 'name': 'snp1'}, {'tenant': self.ctx.tenant_id, 'status': 'IN_PROGRESS', 'stack_id': stack1.id, 'name': 'snp1'}, {'tenant': self.ctx.tenant_id, 'status': 'IN_PROGRESS', 'stack_id': stack2.id, 'name': 'snp2'}]
        for val in values:
            self.assertIsNotNone(db_api.snapshot_create(self.ctx, val))
        self.assertEqual(2, db_api.snapshot_count_all_by_stack(self.ctx, stack1.id))
        self.assertEqual(1, db_api.snapshot_count_all_by_stack(self.ctx, stack2.id))
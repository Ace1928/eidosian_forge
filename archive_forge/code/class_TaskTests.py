import copy
import datetime
import functools
from unittest import mock
import uuid
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import utils as sqlalchemyutils
from sqlalchemy import sql
from glance.common import exception
from glance.common import timeutils
from glance import context
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import models
from glance.tests import functional
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
class TaskTests(test_utils.BaseTestCase):

    def setUp(self):
        super(TaskTests, self).setUp()
        self.admin_id = 'admin'
        self.owner_id = 'user'
        self.adm_context = context.RequestContext(is_admin=True, auth_token='user:admin:admin', tenant=self.admin_id)
        self.context = context.RequestContext(is_admin=False, auth_token='user:user:user', user=self.owner_id)
        self.db_api = db_tests.get_db(self.config)
        self.fixtures = self.build_task_fixtures()
        db_tests.reset_db(self.db_api)

    def build_task_fixtures(self):
        self.context.project_id = str(uuid.uuid4())
        fixtures = [{'owner': self.context.owner, 'type': 'import', 'input': {'import_from': 'file:///a.img', 'import_from_format': 'qcow2', 'image_properties': {'name': 'GreatStack 1.22', 'tags': ['lamp', 'custom']}}}, {'owner': self.context.owner, 'type': 'import', 'input': {'import_from': 'file:///b.img', 'import_from_format': 'qcow2', 'image_properties': {'name': 'GreatStack 1.23', 'tags': ['lamp', 'good']}}}, {'owner': self.context.owner, 'type': 'export', 'input': {'export_uuid': 'deadbeef-dead-dead-dead-beefbeefbeef', 'export_to': 'swift://cloud.foo/myaccount/mycontainer/path', 'export_format': 'qcow2'}}]
        return [build_task_fixture(**fixture) for fixture in fixtures]

    def test_task_get_all_with_filter(self):
        for fixture in self.fixtures:
            self.db_api.task_create(self.adm_context, build_task_fixture(**fixture))
        import_tasks = self.db_api.task_get_all(self.adm_context, filters={'type': 'import'})
        self.assertTrue(import_tasks)
        self.assertEqual(2, len(import_tasks))
        for task in import_tasks:
            self.assertEqual('import', task['type'])
            self.assertEqual(self.context.owner, task['owner'])

    def test_task_get_all_as_admin(self):
        tasks = []
        for fixture in self.fixtures:
            task = self.db_api.task_create(self.adm_context, build_task_fixture(**fixture))
            tasks.append(task)
        import_tasks = self.db_api.task_get_all(self.adm_context)
        self.assertTrue(import_tasks)
        self.assertEqual(3, len(import_tasks))

    def test_task_get_all_marker(self):
        for fixture in self.fixtures:
            self.db_api.task_create(self.adm_context, build_task_fixture(**fixture))
        tasks = self.db_api.task_get_all(self.adm_context, sort_key='id')
        task_ids = [t['id'] for t in tasks]
        tasks = self.db_api.task_get_all(self.adm_context, sort_key='id', marker=task_ids[0])
        self.assertEqual(2, len(tasks))

    def test_task_get_all_limit(self):
        for fixture in self.fixtures:
            self.db_api.task_create(self.adm_context, build_task_fixture(**fixture))
        tasks = self.db_api.task_get_all(self.adm_context, limit=2)
        self.assertEqual(2, len(tasks))
        tasks = self.db_api.task_get_all(self.adm_context, limit=None)
        self.assertEqual(3, len(tasks))
        tasks = self.db_api.task_get_all(self.adm_context, limit=0)
        self.assertEqual(0, len(tasks))

    def test_task_get_all_owned(self):
        then = timeutils.utcnow() + datetime.timedelta(days=365)
        TENANT1 = str(uuid.uuid4())
        ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
        task_values = {'type': 'import', 'status': 'pending', 'input': '{"loc": "fake"}', 'owner': TENANT1, 'expires_at': then}
        self.db_api.task_create(ctxt1, task_values)
        TENANT2 = str(uuid.uuid4())
        ctxt2 = context.RequestContext(is_admin=False, tenant=TENANT2, auth_token='user:%s:user' % TENANT2)
        task_values = {'type': 'export', 'status': 'pending', 'input': '{"loc": "fake"}', 'owner': TENANT2, 'expires_at': then}
        self.db_api.task_create(ctxt2, task_values)
        tasks = self.db_api.task_get_all(ctxt1)
        task_owners = set([task['owner'] for task in tasks])
        expected = set([TENANT1])
        self.assertEqual(sorted(expected), sorted(task_owners))

    def test_task_get(self):
        expires_at = timeutils.utcnow()
        image_id = str(uuid.uuid4())
        fixture = {'owner': self.context.owner, 'type': 'import', 'status': 'pending', 'input': '{"loc": "fake"}', 'result': "{'image_id': %s}" % image_id, 'message': 'blah', 'expires_at': expires_at}
        task = self.db_api.task_create(self.adm_context, fixture)
        self.assertIsNotNone(task)
        self.assertIsNotNone(task['id'])
        task_id = task['id']
        task = self.db_api.task_get(self.adm_context, task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task_id, task['id'])
        self.assertEqual(self.context.owner, task['owner'])
        self.assertEqual('import', task['type'])
        self.assertEqual('pending', task['status'])
        self.assertEqual(fixture['input'], task['input'])
        self.assertEqual(fixture['result'], task['result'])
        self.assertEqual(fixture['message'], task['message'])
        self.assertEqual(expires_at, task['expires_at'])

    def _test_task_get_by_image(self, expired=False, deleted=False, other_owner=False):
        expires_at = timeutils.utcnow()
        if expired is False:
            expires_at += datetime.timedelta(hours=1)
        elif expired is None:
            expires_at = None
        image_id = str(uuid.uuid4())
        fixture = {'owner': other_owner and 'notme!' or self.context.owner, 'type': 'import', 'status': 'pending', 'input': '{"loc": "fake"}', 'result': "{'image_id': %s}" % image_id, 'message': 'blah', 'expires_at': expires_at, 'image_id': image_id, 'user_id': 'me', 'request_id': 'reqid'}
        new_task = self.db_api.task_create(self.adm_context, fixture)
        if deleted:
            self.db_api.task_delete(self.context, new_task['id'])
        return (new_task['id'], self.db_api.tasks_get_by_image(self.context, image_id))

    def test_task_get_by_image_not_expired(self):
        task_id, tasks = self._test_task_get_by_image(expired=False)
        self.assertEqual(1, len(tasks))
        self.assertEqual(task_id, tasks[0]['id'])

    def test_task_get_by_image_expired(self):
        task_id, tasks = self._test_task_get_by_image(expired=True)
        self.assertEqual(0, len(tasks))
        tasks = self.db_api.task_get_all(self.adm_context)
        self.assertEqual(1, len(tasks))
        self.assertEqual(task_id, tasks[0]['id'])
        self.assertTrue(tasks[0]['deleted'])

    def test_task_get_by_image_no_expiry(self):
        task_id, tasks = self._test_task_get_by_image(expired=None)
        self.assertEqual(1, len(tasks))
        tasks = self.db_api.task_get_all(self.adm_context)
        self.assertEqual(1, len(tasks))
        self.assertEqual(task_id, tasks[0]['id'])
        self.assertFalse(tasks[0]['deleted'])
        self.assertIsNone(tasks[0]['expires_at'])

    def test_task_get_by_image_deleted(self):
        task_id, tasks = self._test_task_get_by_image(deleted=True)
        self.assertEqual(0, len(tasks))

    def test_task_get_by_image_not_mine(self):
        task_id, tasks = self._test_task_get_by_image(other_owner=True)
        self.assertEqual(0, len(tasks))

    def test_task_get_all(self):
        now = timeutils.utcnow()
        then = now + datetime.timedelta(days=365)
        image_id = str(uuid.uuid4())
        fixture1 = {'owner': self.context.owner, 'type': 'import', 'status': 'pending', 'input': '{"loc": "fake_1"}', 'result': "{'image_id': %s}" % image_id, 'message': 'blah_1', 'expires_at': then, 'created_at': now, 'updated_at': now}
        fixture2 = {'owner': self.context.owner, 'type': 'import', 'status': 'pending', 'input': '{"loc": "fake_2"}', 'result': "{'image_id': %s}" % image_id, 'message': 'blah_2', 'expires_at': then, 'created_at': now, 'updated_at': now}
        task1 = self.db_api.task_create(self.adm_context, fixture1)
        task2 = self.db_api.task_create(self.adm_context, fixture2)
        self.assertIsNotNone(task1)
        self.assertIsNotNone(task2)
        task1_id = task1['id']
        task2_id = task2['id']
        task_fixtures = {task1_id: fixture1, task2_id: fixture2}
        tasks = self.db_api.task_get_all(self.adm_context)
        self.assertEqual(2, len(tasks))
        self.assertEqual(set((tasks[0]['id'], tasks[1]['id'])), set((task1_id, task2_id)))
        for task in tasks:
            fixture = task_fixtures[task['id']]
            self.assertEqual(self.context.owner, task['owner'])
            self.assertEqual(fixture['type'], task['type'])
            self.assertEqual(fixture['status'], task['status'])
            self.assertEqual(fixture['expires_at'], task['expires_at'])
            self.assertFalse(task['deleted'])
            self.assertIsNone(task['deleted_at'])
            self.assertEqual(fixture['created_at'], task['created_at'])
            self.assertEqual(fixture['updated_at'], task['updated_at'])
            task_details_keys = ['input', 'message', 'result']
            for key in task_details_keys:
                self.assertNotIn(key, task)

    def test_task_soft_delete(self):
        now = timeutils.utcnow()
        then = now + datetime.timedelta(days=365)
        fixture1 = build_task_fixture(id='1', expires_at=now, owner=self.adm_context.owner)
        fixture2 = build_task_fixture(id='2', expires_at=now, owner=self.adm_context.owner)
        fixture3 = build_task_fixture(id='3', expires_at=then, owner=self.adm_context.owner)
        fixture4 = build_task_fixture(id='4', expires_at=then, owner=self.adm_context.owner)
        task1 = self.db_api.task_create(self.adm_context, fixture1)
        task2 = self.db_api.task_create(self.adm_context, fixture2)
        task3 = self.db_api.task_create(self.adm_context, fixture3)
        task4 = self.db_api.task_create(self.adm_context, fixture4)
        self.assertIsNotNone(task1)
        self.assertIsNotNone(task2)
        self.assertIsNotNone(task3)
        self.assertIsNotNone(task4)
        tasks = self.db_api.task_get_all(self.adm_context, sort_key='id', sort_dir='asc')
        self.assertEqual(4, len(tasks))
        self.assertTrue(tasks[0]['deleted'])
        self.assertTrue(tasks[1]['deleted'])
        self.assertFalse(tasks[2]['deleted'])
        self.assertFalse(tasks[3]['deleted'])

    def test_task_create(self):
        task_id = str(uuid.uuid4())
        self.context.project_id = self.context.owner
        values = {'id': task_id, 'owner': self.context.owner, 'type': 'export', 'status': 'pending'}
        task_values = build_task_fixture(**values)
        task = self.db_api.task_create(self.adm_context, task_values)
        self.assertIsNotNone(task)
        self.assertEqual(task_id, task['id'])
        self.assertEqual(self.context.owner, task['owner'])
        self.assertEqual('export', task['type'])
        self.assertEqual('pending', task['status'])
        self.assertEqual({'ping': 'pong'}, task['input'])

    def test_task_create_with_all_task_info_null(self):
        task_id = str(uuid.uuid4())
        self.context.project_id = str(uuid.uuid4())
        values = {'id': task_id, 'owner': self.context.owner, 'type': 'export', 'status': 'pending', 'input': None, 'result': None, 'message': None}
        task_values = build_task_fixture(**values)
        task = self.db_api.task_create(self.adm_context, task_values)
        self.assertIsNotNone(task)
        self.assertEqual(task_id, task['id'])
        self.assertEqual(self.context.owner, task['owner'])
        self.assertEqual('export', task['type'])
        self.assertEqual('pending', task['status'])
        self.assertIsNone(task['input'])
        self.assertIsNone(task['result'])
        self.assertIsNone(task['message'])

    def test_task_update(self):
        self.context.project_id = str(uuid.uuid4())
        result = {'foo': 'bar'}
        task_values = build_task_fixture(owner=self.context.owner, result=result)
        task = self.db_api.task_create(self.adm_context, task_values)
        task_id = task['id']
        fixture = {'status': 'processing', 'message': 'This is a error string'}
        self.delay_inaccurate_clock()
        task = self.db_api.task_update(self.adm_context, task_id, fixture)
        self.assertEqual(task_id, task['id'])
        self.assertEqual(self.context.owner, task['owner'])
        self.assertEqual('import', task['type'])
        self.assertEqual('processing', task['status'])
        self.assertEqual({'ping': 'pong'}, task['input'])
        self.assertEqual(result, task['result'])
        self.assertEqual('This is a error string', task['message'])
        self.assertFalse(task['deleted'])
        self.assertIsNone(task['deleted_at'])
        self.assertIsNone(task['expires_at'])
        self.assertEqual(task_values['created_at'], task['created_at'])
        self.assertGreater(task['updated_at'], task['created_at'])

    def test_task_update_with_all_task_info_null(self):
        self.context.project_id = str(uuid.uuid4())
        task_values = build_task_fixture(owner=self.context.owner, input=None, result=None, message=None)
        task = self.db_api.task_create(self.adm_context, task_values)
        task_id = task['id']
        fixture = {'status': 'processing'}
        self.delay_inaccurate_clock()
        task = self.db_api.task_update(self.adm_context, task_id, fixture)
        self.assertEqual(task_id, task['id'])
        self.assertEqual(self.context.owner, task['owner'])
        self.assertEqual('import', task['type'])
        self.assertEqual('processing', task['status'])
        self.assertIsNone(task['input'])
        self.assertIsNone(task['result'])
        self.assertIsNone(task['message'])
        self.assertFalse(task['deleted'])
        self.assertIsNone(task['deleted_at'])
        self.assertIsNone(task['expires_at'])
        self.assertEqual(task_values['created_at'], task['created_at'])
        self.assertGreater(task['updated_at'], task['created_at'])

    def test_task_delete(self):
        task_values = build_task_fixture(owner=self.context.owner)
        task = self.db_api.task_create(self.adm_context, task_values)
        self.assertIsNotNone(task)
        self.assertFalse(task['deleted'])
        self.assertIsNone(task['deleted_at'])
        task_id = task['id']
        self.db_api.task_delete(self.adm_context, task_id)
        self.assertRaises(exception.TaskNotFound, self.db_api.task_get, self.context, task_id)

    def test_task_delete_as_admin(self):
        task_values = build_task_fixture(owner=self.context.owner)
        task = self.db_api.task_create(self.adm_context, task_values)
        self.assertIsNotNone(task)
        self.assertFalse(task['deleted'])
        self.assertIsNone(task['deleted_at'])
        task_id = task['id']
        self.db_api.task_delete(self.adm_context, task_id)
        del_task = self.db_api.task_get(self.adm_context, task_id, force_show_deleted=True)
        self.assertIsNotNone(del_task)
        self.assertEqual(task_id, del_task['id'])
        self.assertTrue(del_task['deleted'])
        self.assertIsNotNone(del_task['deleted_at'])
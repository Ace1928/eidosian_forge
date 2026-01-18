import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_utils import encodeutils
from oslo_utils.fixture import uuidsentinel as uuids
from oslo_utils import timeutils
from sqlalchemy import orm as sa_orm
from glance.common import crypt
from glance.common import exception
import glance.context
import glance.db
from glance.db.sqlalchemy import api
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class TestTaskRepo(test_utils.BaseTestCase):

    def setUp(self):
        super(TestTaskRepo, self).setUp()
        self.db = unit_test_utils.FakeDB(initialize=False)
        self.context = glance.context.RequestContext(user=USER1, tenant=TENANT1)
        self.task_repo = glance.db.TaskRepo(self.context, self.db)
        self.task_factory = glance.domain.TaskFactory()
        self.fake_task_input = '{"import_from": "swift://cloud.foo/account/mycontainer/path","import_from_format": "qcow2"}'
        self._create_tasks()

    def _create_tasks(self):
        self.tasks = [_db_task_fixture(UUID1, type='import', status='pending', input=self.fake_task_input, result='', owner=TENANT1, message=''), _db_task_fixture(UUID2, type='import', status='processing', input=self.fake_task_input, result='', owner=TENANT1, message=''), _db_task_fixture(UUID3, type='import', status='failure', input=self.fake_task_input, result='', owner=TENANT1, message=''), _db_task_fixture(UUID4, type='import', status='success', input=self.fake_task_input, result='', owner=TENANT2, message='')]
        [self.db.task_create(None, task) for task in self.tasks]

    def test_get(self):
        task = self.task_repo.get(UUID1)
        self.assertEqual(task.task_id, UUID1)
        self.assertEqual('import', task.type)
        self.assertEqual('pending', task.status)
        self.assertEqual(task.task_input, self.fake_task_input)
        self.assertEqual('', task.result)
        self.assertEqual('', task.message)
        self.assertEqual(task.owner, TENANT1)

    def test_get_not_found(self):
        self.assertRaises(exception.NotFound, self.task_repo.get, str(uuid.uuid4()))

    def test_get_forbidden(self):
        self.assertRaises(exception.NotFound, self.task_repo.get, UUID4)

    def test_list(self):
        tasks = self.task_repo.list()
        task_ids = set([i.task_id for i in tasks])
        self.assertEqual(set([UUID1, UUID2, UUID3]), task_ids)

    def test_list_with_type(self):
        filters = {'type': 'import'}
        tasks = self.task_repo.list(filters=filters)
        task_ids = set([i.task_id for i in tasks])
        self.assertEqual(set([UUID1, UUID2, UUID3]), task_ids)

    def test_list_with_status(self):
        filters = {'status': 'failure'}
        tasks = self.task_repo.list(filters=filters)
        task_ids = set([i.task_id for i in tasks])
        self.assertEqual(set([UUID3]), task_ids)

    def test_list_with_marker(self):
        full_tasks = self.task_repo.list()
        full_ids = [i.task_id for i in full_tasks]
        marked_tasks = self.task_repo.list(marker=full_ids[0])
        actual_ids = [i.task_id for i in marked_tasks]
        self.assertEqual(full_ids[1:], actual_ids)

    def test_list_with_last_marker(self):
        tasks = self.task_repo.list()
        marked_tasks = self.task_repo.list(marker=tasks[-1].task_id)
        self.assertEqual(0, len(marked_tasks))

    def test_limited_list(self):
        limited_tasks = self.task_repo.list(limit=2)
        self.assertEqual(2, len(limited_tasks))

    def test_list_with_marker_and_limit(self):
        full_tasks = self.task_repo.list()
        full_ids = [i.task_id for i in full_tasks]
        marked_tasks = self.task_repo.list(marker=full_ids[0], limit=1)
        actual_ids = [i.task_id for i in marked_tasks]
        self.assertEqual(full_ids[1:2], actual_ids)

    def test_sorted_list(self):
        tasks = self.task_repo.list(sort_key='status', sort_dir='desc')
        task_ids = [i.task_id for i in tasks]
        self.assertEqual([UUID2, UUID1, UUID3], task_ids)

    def test_add_task(self):
        task_type = 'import'
        image_id = 'fake_image_id'
        user_id = 'fake_user'
        request_id = 'fake_request_id'
        task = self.task_factory.new_task(task_type, None, image_id, user_id, request_id, task_input=self.fake_task_input)
        self.assertEqual(task.updated_at, task.created_at)
        self.task_repo.add(task)
        retrieved_task = self.task_repo.get(task.task_id)
        self.assertEqual(task.updated_at, retrieved_task.updated_at)
        self.assertEqual(self.fake_task_input, retrieved_task.task_input)
        self.assertEqual(image_id, task.image_id)
        self.assertEqual(user_id, task.user_id)
        self.assertEqual(request_id, task.request_id)

    def test_save_task(self):
        task = self.task_repo.get(UUID1)
        original_update_time = task.updated_at
        self.delay_inaccurate_clock()
        self.task_repo.save(task)
        current_update_time = task.updated_at
        self.assertGreater(current_update_time, original_update_time)
        task = self.task_repo.get(UUID1)
        self.assertEqual(current_update_time, task.updated_at)

    def test_remove_task(self):
        task = self.task_repo.get(UUID1)
        self.task_repo.remove(task)
        self.assertRaises(exception.NotFound, self.task_repo.get, task.task_id)
import datetime
from testtools import content as ttc
import time
from unittest import mock
import uuid
from oslo_log import log as logging
from oslo_utils import fixture as time_fixture
from oslo_utils import units
from glance.tests import functional
from glance.tests import utils as test_utils
class TestImageImportLocking(functional.SynchronousAPIBase):

    def _get_image_import_task(self, image_id, task_id=None):
        if task_id is None:
            image = self.api_get('/v2/images/%s' % image_id).json
            task_id = image['os_glance_import_task']
        return self.api_get('/v2/tasks/%s' % task_id).json

    def _test_import_copy(self, warp_time=False):
        self.start_server()
        state = {'want_run': True}
        image_id = self._create_and_import(stores=['store1'])

        def slow_fake_set_data(data_iter, size=None, backend=None, set_active=True):
            me = str(uuid.uuid4())
            while state['want_run'] == True:
                LOG.info('fake_set_data running %s', me)
                state['running'] = True
                time.sleep(0.1)
            LOG.info('fake_set_data ended %s', me)
        tf = time_fixture.TimeFixture()
        self.useFixture(tf)
        with mock.patch('glance.location.ImageProxy.set_data') as mock_sd:
            mock_sd.side_effect = slow_fake_set_data
            resp = self._import_copy(image_id, ['store2'])
            self.addDetail('First import response', ttc.text_content(str(resp)))
            self.assertEqual(202, resp.status_code)
            for i in range(0, 10):
                if 'running' in state:
                    break
                time.sleep(0.1)
        self.assertTrue(state.get('running', False), 'slow_fake_set_data() never ran')
        first_import_task = self._get_image_import_task(image_id)
        self.assertEqual('processing', first_import_task['status'])
        if warp_time:
            tf.advance_time_delta(datetime.timedelta(hours=2))
        resp = self._import_copy(image_id, ['store3'])
        time.sleep(0.1)
        self.addDetail('Second import response', ttc.text_content(str(resp)))
        if warp_time:
            self.assertEqual(202, resp.status_code)
        else:
            self.assertEqual(409, resp.status_code)
        self.addDetail('First task', ttc.text_content(str(first_import_task)))
        second_import_task = self._get_image_import_task(image_id)
        first_import_task = self._get_image_import_task(image_id, first_import_task['id'])
        if warp_time:
            self.assertNotEqual(first_import_task['id'], second_import_task['id'])
            self.assertEqual('failure', first_import_task['status'])
            self.assertEqual('Expired lock preempted', first_import_task['message'])
            self.assertEqual('processing', second_import_task['status'])
        else:
            self.assertEqual(first_import_task['id'], second_import_task['id'])
        return (image_id, state)

    def test_import_copy_locked(self):
        self._test_import_copy(warp_time=False)

    def test_import_copy_bust_lock(self):
        image_id, state = self._test_import_copy(warp_time=True)
        for i in range(0, 10):
            image = self.api_get('/v2/images/%s' % image_id).json
            if image['stores'] == 'store1,store3':
                break
            time.sleep(0.1)
        image = self.api_get('/v2/images/%s' % image_id).json
        self.assertEqual('store1,store3', image['stores'])
        self.assertEqual('', image['os_glance_failed_import'])
        state['want_run'] = False
        for i in range(0, 10):
            image = self.api_get('/v2/images/%s' % image_id).json
            time.sleep(0.1)
        image = self.api_get('/v2/images/%s' % image_id).json
        self.assertEqual('', image.get('os_glance_import_task', ''))
        self.assertEqual('', image['os_glance_importing_to_stores'])
        self.assertEqual('', image['os_glance_failed_import'])
        self.assertEqual('store1,store3', image['stores'])

    @mock.patch('oslo_utils.timeutils.StopWatch.expired', new=lambda x: True)
    def test_import_task_status(self):
        self.start_server()
        limit = 3 * units.Mi
        image_id = self._create_and_stage(data_iter=test_utils.FakeData(limit))
        statuses = []

        def grab_task_status():
            image = self.api_get('/v2/images/%s' % image_id).json
            task_id = image['os_glance_import_task']
            task = self.api_get('/v2/tasks/%s' % task_id).json
            msg = task['message']
            if msg not in statuses:
                statuses.append(msg)

        def fake_upload(data, *a, **k):
            while True:
                grab_task_status()
                if not data.read(65536):
                    break
                time.sleep(0.1)
        with mock.patch('glance.location.ImageProxy._upload_to_store') as mu:
            mu.side_effect = fake_upload
            resp = self._import_direct(image_id, ['store2'])
            self.assertEqual(202, resp.status_code)
            for i in range(0, 100):
                image = self.api_get('/v2/images/%s' % image_id).json
                if not image.get('os_glance_import_task'):
                    break
                time.sleep(0.1)
        self.assertEqual('active', image['status'])
        self.assertEqual(['', 'Copied 0 MiB', 'Copied 1 MiB', 'Copied 2 MiB', 'Copied 3 MiB'], statuses)
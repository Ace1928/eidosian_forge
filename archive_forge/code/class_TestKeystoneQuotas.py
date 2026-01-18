import hashlib
import http.client as http
import os
import subprocess
import tempfile
import time
import urllib
import uuid
import fixtures
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_serialization import jsonutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests
from glance.quota import keystone as ks_quota
from glance.tests import functional
from glance.tests.functional import ft_utils as func_utils
from glance.tests import utils as test_utils
class TestKeystoneQuotas(functional.SynchronousAPIBase):

    def setUp(self):
        super(TestKeystoneQuotas, self).setUp()
        self.config(endpoint_id='ENDPOINT_ID', group='oslo_limit')
        self.config(use_keystone_limits=True)
        self.config(filesystem_store_datadir='/tmp/foo', group='os_glance_tasks_store')
        self.enforcer_mock = self.useFixture(fixtures.MockPatchObject(ks_quota, 'limit')).mock

    def set_limit(self, limits):
        self.enforcer_mock.Enforcer = get_enforcer_class(limits)

    def test_upload(self):
        self.set_limit({'image_size_total': 5, 'image_count_total': 10, 'image_count_uploading': 10})
        self.start_server()
        image_id = self._create_and_upload(data_iter=test_utils.FakeData(3 * units.Mi))
        self._create_and_upload(data_iter=test_utils.FakeData(3 * units.Mi))
        self._create_and_upload(expected_code=413)
        self.api_delete('/v2/images/%s' % image_id)
        self._create_and_upload()

    def test_import(self):
        self.set_limit({'image_size_total': 5, 'image_count_total': 10, 'image_count_uploading': 10})
        self.start_server()
        image_id = self._create_and_upload(data_iter=test_utils.FakeData(3 * units.Mi))
        self._create_and_upload(data_iter=test_utils.FakeData(3 * units.Mi))
        self._create_and_import(stores=['store1'], expected_code=413)
        self.api_delete('/v2/images/%s' % image_id)
        self._create_and_import(stores=['store1'])

    def test_import_would_go_over(self):
        self.set_limit({'image_size_total': 5, 'image_count_total': 10, 'image_count_uploading': 10})
        self.start_server()
        image_id = self._create_and_upload(data_iter=test_utils.FakeData(3 * units.Mi))
        import_id = self._create_and_stage(data_iter=test_utils.FakeData(3 * units.Mi))
        self._import_direct(import_id, ['store1'])
        image = self._wait_for_import(import_id)
        task = self._get_latest_task(import_id)
        self.assertEqual('failure', task['status'])
        self.assertIn('image_size_total is over limit of 5 due to current usage 3 and delta 3', task['message'])
        resp = self.api_delete('/v2/images/%s' % image_id)
        self.assertEqual(204, resp.status_code)
        import_id = self._create_and_stage(data_iter=test_utils.FakeData(3 * units.Mi))
        resp = self._import_direct(import_id, ['store1'])
        self.assertEqual(202, resp.status_code)
        image = self._wait_for_import(import_id)
        self.assertEqual('active', image['status'])
        task = self._get_latest_task(import_id)
        self.assertEqual('success', task['status'])

    def test_copy(self):
        self.set_limit({'image_size_total': 5, 'image_count_total': 10, 'image_stage_total': 15, 'image_count_uploading': 10})
        self.start_server()
        image_id = self._create_and_import(stores=['store1'], data_iter=test_utils.FakeData(3 * units.Mi))
        req = self._import_copy(image_id, ['store2'])
        self.assertEqual(202, req.status_code)
        self._wait_for_import(image_id)
        self.assertEqual('success', self._get_latest_task(image_id)['status'])
        req = self._import_copy(image_id, ['store3'])
        self.assertEqual(413, req.status_code)
        self.set_limit({'image_size_total': 15, 'image_count_total': 10, 'image_stage_total': 5, 'image_count_uploading': 10})
        req = self._import_copy(image_id, ['store3'])
        self.assertEqual(202, req.status_code)
        self._wait_for_import(image_id)
        self.assertEqual('failure', self._get_latest_task(image_id)['status'])
        self.set_limit({'image_size_total': 15, 'image_count_total': 10, 'image_stage_total': 10, 'image_count_uploading': 10})
        req = self._import_copy(image_id, ['store3'])
        self.assertEqual(202, req.status_code)
        self._wait_for_import(image_id)
        self.assertEqual('success', self._get_latest_task(image_id)['status'])

    def test_stage(self):
        self.set_limit({'image_size_total': 15, 'image_stage_total': 5, 'image_count_total': 10, 'image_count_uploading': 10})
        self.start_server()
        image_id = self._create_and_stage(data_iter=test_utils.FakeData(6 * units.Mi))
        self._create_and_stage(expected_code=413)
        image_id2 = self._create().json['id']
        req = self._import_web_download(image_id2, ['store1'], 'http://example.com/foo.img')
        self.assertEqual(202, req.status_code)
        self._wait_for_import(image_id2)
        task = self._get_latest_task(image_id2)
        self.assertEqual('failure', task['status'])
        self.assertIn('image_stage_total is over limit', task['message'])
        req = self._import_direct(image_id, ['store1'])
        self.assertEqual(202, req.status_code)
        self._wait_for_import(image_id)
        self._create_and_stage(data_iter=test_utils.FakeData(6 * units.Mi))

    def test_create(self):
        self.set_limit({'image_size_total': 15, 'image_count_total': 2, 'image_count_uploading': 10})
        self.start_server()
        image_id = self._create().json['id']
        self._create()
        resp = self._create()
        self.assertEqual(413, resp.status_code)
        self.api_delete('/v2/images/%s' % image_id)
        self._create()

    def test_uploading_methods(self):
        self.set_limit({'image_size_total': 100, 'image_stage_total': 100, 'image_count_total': 100, 'image_count_uploading': 1})
        self.start_server()
        image_id = self._create_and_stage()
        self._create_and_stage(expected_code=413)
        self._create_and_upload(expected_code=413)
        resp = self._import_direct(image_id, ['store1'])
        self.assertEqual(202, resp.status_code)
        self.assertEqual('active', self._wait_for_import(image_id)['status'])
        self._create_and_upload()
        image_id2 = self._create_and_stage()
        resp = self._import_copy(image_id, ['store2'])
        self.assertEqual(202, resp.status_code)
        self._wait_for_import(image_id)
        task = self._get_latest_task(image_id)
        self.assertEqual('failure', task['status'])
        self.assertIn('Resource image_count_uploading is over limit', task['message'])
        self._import_direct(image_id2, ['store1'])
        self.assertEqual(202, resp.status_code)
        self._wait_for_import(image_id2)
        self._create_and_upload()
        resp = self._import_copy(image_id, ['store2'])
        self.assertEqual(202, resp.status_code)
        self._wait_for_import(image_id)
        task = self._get_latest_task(image_id)
        self.assertEqual('success', task['status'])
        self._create_and_upload()
        self._create_and_import(stores=['store1'])
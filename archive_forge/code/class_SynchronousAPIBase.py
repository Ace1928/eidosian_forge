import abc
import atexit
import datetime
import errno
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from testtools import content as ttc
import textwrap
import time
from unittest import mock
import urllib.parse as urlparse
import uuid
import fixtures
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_config import cfg
from oslo_serialization import jsonutils
import testtools
import webob
from glance.common import config
from glance.common import utils
from glance.common import wsgi
from glance.db.sqlalchemy import api as db_api
from glance import tests as glance_tests
from glance.tests import utils as test_utils
import glance.async_
class SynchronousAPIBase(test_utils.BaseTestCase):
    """A base class that provides synchronous calling into the API.

    This provides a way to directly call into the API WSGI stack
    without starting a separate server, and with a simple paste
    pipeline. Configured with multi-store and a real database.

    This differs from the FunctionalTest lineage above in that they
    start a full copy of the API server as a separate process, whereas
    this calls directly into the WSGI stack. This test base is
    appropriate for situations where you need to be able to mock the
    state of the world (i.e. warp time, or inject errors) but should
    not be used for happy-path testing where FunctionalTest provides
    more isolation.

    To use this, inherit and run start_server() before you are ready
    to make API calls (either in your setUp() or per-test if you need
    to change config or mocking).

    Once started, use the api_get(), api_put(), api_post(), and
    api_delete() methods to make calls to the API.

    """
    TENANT = str(uuid.uuid4())

    @mock.patch('oslo_db.sqlalchemy.enginefacade.writer.get_engine')
    def setup_database(self, mock_get_engine):
        """Configure and prepare a fresh sqlite database."""
        db_file = 'sqlite:///%s/test.db' % self.test_dir
        self.config(connection=db_file, group='database')
        db_api.clear_db_env()
        engine = db_api.get_engine()
        mock_get_engine.return_value = engine
        with mock.patch('logging.config'):
            test_utils.db_sync(engine=engine)

    def setup_simple_paste(self):
        """Setup a very simple no-auth paste pipeline.

        This configures the API to be very direct, including only the
        middleware absolutely required for consistent API calls.
        """
        self.paste_config = os.path.join(self.test_dir, 'glance-api-paste.ini')
        with open(self.paste_config, 'w') as f:
            f.write(textwrap.dedent('\n            [filter:context]\n            paste.filter_factory = glance.api.middleware.context:                ContextMiddleware.factory\n            [filter:fakeauth]\n            paste.filter_factory = glance.tests.utils:                FakeAuthMiddleware.factory\n            [filter:cache]\n            paste.filter_factory = glance.api.middleware.cache:            CacheFilter.factory\n            [filter:cachemanage]\n            paste.filter_factory = glance.api.middleware.cache_manage:            CacheManageFilter.factory\n            [pipeline:glance-api-cachemanagement]\n            pipeline = context cache cachemanage rootapp\n            [pipeline:glance-api-caching]\n            pipeline = context cache rootapp\n            [pipeline:glance-api]\n            pipeline = context rootapp\n            [composite:rootapp]\n            paste.composite_factory = glance.api:root_app_factory\n            /v2: apiv2app\n            [app:apiv2app]\n            paste.app_factory = glance.api.v2.router:API.factory\n            '))

    def _store_dir(self, store):
        return os.path.join(self.test_dir, store)

    def setup_stores(self):
        """Configures multiple backend stores.

        This configures the API with three file-backed stores (store1,
        store2, and store3) as well as a os_glance_staging_store for
        imports.

        """
        self.config(enabled_backends={'store1': 'file', 'store2': 'file', 'store3': 'file'})
        glance_store.register_store_opts(CONF, reserved_stores=wsgi.RESERVED_STORES)
        self.config(default_backend='store1', group='glance_store')
        self.config(filesystem_store_datadir=self._store_dir('store1'), group='store1')
        self.config(filesystem_store_datadir=self._store_dir('store2'), group='store2')
        self.config(filesystem_store_datadir=self._store_dir('store3'), group='store3')
        self.config(filesystem_store_datadir=self._store_dir('staging'), group='os_glance_staging_store')
        self.config(filesystem_store_datadir=self._store_dir('tasks'), group='os_glance_tasks_store')
        glance_store.create_multi_stores(CONF, reserved_stores=wsgi.RESERVED_STORES)
        glance_store.verify_store()

    def setUp(self):
        super(SynchronousAPIBase, self).setUp()
        self.setup_database()
        self.setup_simple_paste()
        self.setup_stores()

    def start_server(self, enable_cache=True, set_worker_url=True):
        """Builds and "starts" the API server.

        Note that this doesn't actually "start" anything like
        FunctionalTest does above, but that terminology is used here
        to make it seem like the same sort of pattern.
        """
        config.set_config_defaults()
        root_app = 'glance-api'
        if enable_cache:
            root_app = 'glance-api-cachemanagement'
            self.config(image_cache_dir=self._store_dir('cache'))
        if set_worker_url:
            self.config(worker_self_reference_url='http://workerx')
        self.api = config.load_paste_app(root_app, conf_file=self.paste_config)
        self.config(enforce_new_defaults=True, group='oslo_policy')
        self.config(enforce_scope=True, group='oslo_policy')

    def _headers(self, custom_headers=None):
        base_headers = {'X-Identity-Status': 'Confirmed', 'X-Auth-Token': '932c5c84-02ac-4fe5-a9ba-620af0e2bb96', 'X-User-Id': 'f9a41d13-0c13-47e9-bee2-ce4e8bfe958e', 'X-Tenant-Id': self.TENANT, 'Content-Type': 'application/json', 'X-Roles': 'admin'}
        base_headers.update(custom_headers or {})
        return base_headers

    def api_request(self, method, url, headers=None, data=None, json=None, body_file=None):
        """Perform a request against the API.

        NOTE: Most code should use api_get(), api_post(), api_put(),
              or api_delete() instead!

        :param method: The HTTP method to use (i.e. GET, POST, etc)
        :param url: The *path* part of the URL to call (i.e. /v2/images)
        :param headers: Optional updates to the default set of headers
        :param data: Optional bytes data payload to send (overrides @json)
        :param json: Optional dict structure to be jsonified and sent as
                     the payload (mutually exclusive with @data)
        :param body_file: Optional io.IOBase to provide as the input data
                          stream for the request (overrides @data)
        :returns: A webob.Response object
        """
        headers = self._headers(headers)
        req = webob.Request.blank(url, method=method, headers=headers)
        if json and (not data):
            data = jsonutils.dumps(json).encode()
        if data and (not body_file):
            req.body = data
        elif body_file:
            req.body_file = body_file
        return self.api(req)

    def api_get(self, url, headers=None):
        """Perform a GET request against the API.

        :param url: The *path* part of the URL to call (i.e. /v2/images)
        :param headers: Optional updates to the default set of headers
        :returns: A webob.Response object
        """
        return self.api_request('GET', url, headers=headers)

    def api_post(self, url, headers=None, data=None, json=None, body_file=None):
        """Perform a POST request against the API.

        :param url: The *path* part of the URL to call (i.e. /v2/images)
        :param headers: Optional updates to the default set of headers
        :param data: Optional bytes data payload to send (overrides @json)
        :param json: Optional dict structure to be jsonified and sent as
                     the payload (mutually exclusive with @data)
        :param body_file: Optional io.IOBase to provide as the input data
                          stream for the request (overrides @data)
        :returns: A webob.Response object
        """
        return self.api_request('POST', url, headers=headers, data=data, json=json, body_file=body_file)

    def api_put(self, url, headers=None, data=None, json=None, body_file=None):
        """Perform a PUT request against the API.

        :param url: The *path* part of the URL to call (i.e. /v2/images)
        :param headers: Optional updates to the default set of headers
        :param data: Optional bytes data payload to send (overrides @json,
                     mutually exclusive with body_file)
        :param json: Optional dict structure to be jsonified and sent as
                     the payload (mutually exclusive with @data)
        :param body_file: Optional io.IOBase to provide as the input data
                          stream for the request (overrides @data)
        :returns: A webob.Response object
        """
        return self.api_request('PUT', url, headers=headers, data=data, json=json, body_file=body_file)

    def api_delete(self, url, headers=None):
        """Perform a DELETE request against the API.

        :param url: The *path* part of the URL to call (i.e. /v2/images)
        :param headers: Optional updates to the default set of headers
        :returns: A webob.Response object
        """
        return self.api_request('DELETE', url, headers=headers)

    def api_patch(self, url, *patches, headers=None):
        """Perform a PATCH request against the API.

        :param url: The *path* part of the URL to call (i.e. /v2/images)
        :param patches: One or more patch dicts
        :param headers: Optional updates to the default set of headers
        :returns: A webob.Response object
        """
        if not headers:
            headers = {}
        headers['Content-Type'] = 'application/openstack-images-v2.1-json-patch'
        return self.api_request('PATCH', url, headers=headers, json=list(patches))

    def _import_copy(self, image_id, stores, headers=None):
        """Do an import of image_id to the given stores."""
        body = {'method': {'name': 'copy-image'}, 'stores': stores, 'all_stores': False}
        return self.api_post('/v2/images/%s/import' % image_id, headers=headers, json=body)

    def _import_direct(self, image_id, stores, headers=None):
        """Do an import of image_id to the given stores."""
        body = {'method': {'name': 'glance-direct'}, 'stores': stores, 'all_stores': False}
        return self.api_post('/v2/images/%s/import' % image_id, headers=headers, json=body)

    def _import_web_download(self, image_id, stores, url, headers=None):
        """Do an import of image_id to the given stores."""
        body = {'method': {'name': 'web-download', 'uri': url}, 'stores': stores, 'all_stores': False}
        return self.api_post('/v2/images/%s/import' % image_id, headers=headers, json=body)

    def _create_and_upload(self, data_iter=None, expected_code=204, visibility=None):
        data = {'name': 'foo', 'container_format': 'bare', 'disk_format': 'raw'}
        if visibility:
            data['visibility'] = visibility
        resp = self.api_post('/v2/images', json=data)
        self.assertEqual(201, resp.status_code, resp.text)
        image = jsonutils.loads(resp.text)
        if data_iter:
            resp = self.api_put('/v2/images/%s/file' % image['id'], headers={'Content-Type': 'application/octet-stream'}, body_file=data_iter)
        else:
            resp = self.api_put('/v2/images/%s/file' % image['id'], headers={'Content-Type': 'application/octet-stream'}, data=b'IMAGEDATA')
        self.assertEqual(expected_code, resp.status_code)
        return image['id']

    def _create_and_stage(self, data_iter=None, expected_code=204, visibility=None, extra={}):
        data = {'name': 'foo', 'container_format': 'bare', 'disk_format': 'raw'}
        if visibility:
            data['visibility'] = visibility
        data.update(extra)
        resp = self.api_post('/v2/images', json=data)
        image = jsonutils.loads(resp.text)
        if data_iter:
            resp = self.api_put('/v2/images/%s/stage' % image['id'], headers={'Content-Type': 'application/octet-stream'}, body_file=data_iter)
        else:
            resp = self.api_put('/v2/images/%s/stage' % image['id'], headers={'Content-Type': 'application/octet-stream'}, data=b'IMAGEDATA')
        self.assertEqual(expected_code, resp.status_code)
        return image['id']

    def _wait_for_import(self, image_id, retries=10):
        for i in range(0, retries):
            image = self.api_get('/v2/images/%s' % image_id).json
            if not image.get('os_glance_import_task'):
                break
            self.addDetail('Create-Import task id', ttc.text_content(image['os_glance_import_task']))
            time.sleep(1)
        self.assertIsNone(image.get('os_glance_import_task'), 'Timed out waiting for task to complete')
        return image

    def _create_and_import(self, stores=[], data_iter=None, expected_code=202, visibility=None, extra={}):
        """Create an image, stage data, and import into the given stores.

        :returns: image_id
        """
        image_id = self._create_and_stage(data_iter=data_iter, visibility=visibility, extra=extra)
        resp = self._import_direct(image_id, stores)
        self.assertEqual(expected_code, resp.status_code)
        if expected_code >= 400:
            return image_id
        image = self._wait_for_import(image_id)
        self.assertEqual('active', image['status'])
        return image_id

    def _get_latest_task(self, image_id):
        tasks = self.api_get('/v2/images/%s/tasks' % image_id).json['tasks']
        tasks = sorted(tasks, key=lambda t: t['updated_at'])
        self.assertGreater(len(tasks), 0)
        return tasks[-1]

    def _create(self):
        return self.api_post('/v2/images', json={'name': 'foo', 'container_format': 'bare', 'disk_format': 'raw'})

    def _create_metadef_resource(self, path=None, data=None, expected_code=201):
        resp = self.api_post(path, json=data)
        md_resource = jsonutils.loads(resp.text)
        self.assertEqual(expected_code, resp.status_code)
        return md_resource
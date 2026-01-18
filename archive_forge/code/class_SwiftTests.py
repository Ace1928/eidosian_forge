import copy
from unittest import mock
import fixtures
import hashlib
import http.client
import importlib
import io
import tempfile
import uuid
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests_mock
import swiftclient
from glance_store._drivers.swift import connection_manager as manager
from glance_store._drivers.swift import store as swift
from glance_store._drivers.swift import utils as sutils
from glance_store import capabilities
from glance_store import exceptions
from glance_store import location
import glance_store.multi_backend as store
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
class SwiftTests(object):

    def mock_keystone_client(self):
        swift.ks_v3 = mock.MagicMock()
        swift.ks_session = mock.MagicMock()
        swift.ks_client = mock.MagicMock()

    def stub_out_swiftclient(self, swift_store_auth_version):
        fixture_containers = ['glance']
        fixture_container_headers = {}
        fixture_headers = {'glance/%s' % FAKE_UUID: {'content-length': FIVE_KB, 'etag': 'c2e5db72bd7fd153f53ede5da5a06de3'}, 'glance/%s' % FAKE_UUID2: {'x-static-large-object': 'true'}}
        fixture_objects = {'glance/%s' % FAKE_UUID: io.BytesIO(b'*' * FIVE_KB), 'glance/%s' % FAKE_UUID2: io.BytesIO(b'*' * FIVE_KB)}

        def fake_head_container(url, token, container, **kwargs):
            if container not in fixture_containers:
                msg = 'No container %s found' % container
                status = http.client.NOT_FOUND
                raise swiftclient.ClientException(msg, http_status=status)
            return fixture_container_headers

        def fake_put_container(url, token, container, **kwargs):
            fixture_containers.append(container)

        def fake_post_container(url, token, container, headers, **kwargs):
            for key, value in headers.items():
                fixture_container_headers[key] = value

        def fake_put_object(url, token, container, name, contents, **kwargs):
            global SWIFT_PUT_OBJECT_CALLS
            SWIFT_PUT_OBJECT_CALLS += 1
            CHUNKSIZE = 64 * units.Ki
            fixture_key = '%s/%s' % (container, name)
            if fixture_key not in fixture_headers:
                if kwargs.get('headers'):
                    manifest = kwargs.get('headers').get('X-Object-Manifest')
                    etag = kwargs.get('headers').get('ETag', md5(b'', usedforsecurity=False).hexdigest())
                    fixture_headers[fixture_key] = {'manifest': True, 'etag': etag, 'x-object-manifest': manifest}
                    fixture_objects[fixture_key] = None
                    return etag
                if hasattr(contents, 'read'):
                    fixture_object = io.BytesIO()
                    read_len = 0
                    chunk = contents.read(CHUNKSIZE)
                    checksum = md5(usedforsecurity=False)
                    while chunk:
                        fixture_object.write(chunk)
                        read_len += len(chunk)
                        checksum.update(chunk)
                        chunk = contents.read(CHUNKSIZE)
                    etag = checksum.hexdigest()
                else:
                    fixture_object = io.BytesIO(contents)
                    read_len = len(contents)
                    etag = md5(fixture_object.getvalue(), usedforsecurity=False).hexdigest()
                if read_len > MAX_SWIFT_OBJECT_SIZE:
                    msg = 'Image size:%d exceeds Swift max:%d' % (read_len, MAX_SWIFT_OBJECT_SIZE)
                    raise swiftclient.ClientException(msg, http_status=http.client.REQUEST_ENTITY_TOO_LARGE)
                fixture_objects[fixture_key] = fixture_object
                fixture_headers[fixture_key] = {'content-length': read_len, 'etag': etag}
                return etag
            else:
                msg = 'Object PUT failed - Object with key %s already exists' % fixture_key
                raise swiftclient.ClientException(msg, http_status=http.client.CONFLICT)

        def fake_get_object(conn, container, name, **kwargs):
            fixture_key = '%s/%s' % (container, name)
            if fixture_key not in fixture_headers:
                msg = 'Object GET failed'
                status = http.client.NOT_FOUND
                raise swiftclient.ClientException(msg, http_status=status)
            byte_range = None
            headers = kwargs.get('headers', dict())
            if headers is not None:
                headers = dict(((k.lower(), v) for k, v in headers.items()))
                if 'range' in headers:
                    byte_range = headers.get('range')
            fixture = fixture_headers[fixture_key]
            if 'manifest' in fixture:
                chunk_keys = sorted([k for k in fixture_headers.keys() if k.startswith(fixture_key) and k != fixture_key])
                result = io.BytesIO()
                for key in chunk_keys:
                    result.write(fixture_objects[key].getvalue())
            else:
                result = fixture_objects[fixture_key]
            if byte_range is not None:
                start = int(byte_range.split('=')[1].strip('-'))
                result = io.BytesIO(result.getvalue()[start:])
                fixture_headers[fixture_key]['content-length'] = len(result.getvalue())
            return (fixture_headers[fixture_key], result)

        def fake_head_object(url, token, container, name, **kwargs):
            try:
                fixture_key = '%s/%s' % (container, name)
                return fixture_headers[fixture_key]
            except KeyError:
                msg = 'Object HEAD failed - Object does not exist'
                status = http.client.NOT_FOUND
                raise swiftclient.ClientException(msg, http_status=status)

        def fake_delete_object(url, token, container, name, **kwargs):
            fixture_key = '%s/%s' % (container, name)
            if fixture_key not in fixture_headers:
                msg = 'Object DELETE failed - Object does not exist'
                status = http.client.NOT_FOUND
                raise swiftclient.ClientException(msg, http_status=status)
            else:
                del fixture_headers[fixture_key]
                del fixture_objects[fixture_key]

        def fake_http_connection(*args, **kwargs):
            return None

        def fake_get_auth(url, user, key, auth_version, **kwargs):
            if url is None:
                return (None, None)
            if 'http' in url and '://' not in url:
                raise ValueError('Invalid url %s' % url)
            if swift_store_auth_version != auth_version:
                msg = 'AUTHENTICATION failed (version mismatch)'
                raise swiftclient.ClientException(msg)
            return (None, None)
        self.useFixture(fixtures.MockPatch('swiftclient.client.head_container', fake_head_container))
        self.useFixture(fixtures.MockPatch('swiftclient.client.put_container', fake_put_container))
        self.useFixture(fixtures.MockPatch('swiftclient.client.post_container', fake_post_container))
        self.useFixture(fixtures.MockPatch('swiftclient.client.put_object', fake_put_object))
        self.useFixture(fixtures.MockPatch('swiftclient.client.delete_object', fake_delete_object))
        self.useFixture(fixtures.MockPatch('swiftclient.client.head_object', fake_head_object))
        self.useFixture(fixtures.MockPatch('swiftclient.client.Connection.get_object', fake_get_object))
        self.useFixture(fixtures.MockPatch('swiftclient.client.get_auth', fake_get_auth))
        self.useFixture(fixtures.MockPatch('swiftclient.client.http_connection', fake_http_connection))

    @property
    def swift_store_user(self):
        return 'tenant:user1'

    def test_get_size(self):
        """
        Test that we can get the size of an object in the swift store
        """
        uri = 'swift://%s:key@auth_address/glance/%s' % (self.swift_store_user, FAKE_UUID)
        loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
        image_size = self.store.get_size(loc)
        self.assertEqual(5120, image_size)

    @mock.patch.object(store, 'get_store_from_store_identifier')
    def test_get_size_with_multi_tenant_on(self, mock_get):
        """Test that single tenant uris work with multi tenant on."""
        mock_get.return_value = self.store
        uri = 'swift://%s:key@auth_address/glance/%s' % (self.swift_store_user, FAKE_UUID)
        self.config(group='swift1', swift_store_config_file=None)
        self.config(group='swift1', swift_store_multi_tenant=True)
        ctxt = mock.MagicMock()
        size = store.get_size_from_uri_and_backend(uri, 'swift1', context=ctxt)
        self.assertEqual(5120, size)

    def test_multi_tenant_with_swift_config(self):
        """
        Test that Glance does not start when a config file is set on
        multi-tenant mode
        """
        schemes = ['swift', 'swift+config']
        for s in schemes:
            self.config(group='glance_store', default_backend='swift1')
            self.config(group='swift1', swift_store_config_file='not/none', swift_store_multi_tenant=True)
            self.assertRaises(exceptions.BadStoreConfiguration, Store, self.conf, backend='swift1')

    def test_get(self):
        """Test a "normal" retrieval of an image in chunks."""
        uri = 'swift://%s:key@auth_address/glance/%s' % (self.swift_store_user, FAKE_UUID)
        loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
        image_swift, image_size = self.store.get(loc)
        self.assertEqual(5120, image_size)
        expected_data = b'*' * FIVE_KB
        data = b''
        for chunk in image_swift:
            data += chunk
        self.assertEqual(expected_data, data)

    def test_get_with_retry(self):
        """
        Test a retrieval where Swift does not get the full image in a single
        request.
        """
        uri = 'swift://%s:key@auth_address/glance/%s' % (self.swift_store_user, FAKE_UUID)
        loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
        ctxt = mock.MagicMock()
        image_swift, image_size = self.store.get(loc, context=ctxt)
        resp_full = b''.join([chunk for chunk in image_swift.wrapped])
        resp_half = resp_full[:len(resp_full) // 2]
        resp_half = io.BytesIO(resp_half)
        manager = self.store.get_manager(loc.store_location, ctxt)
        image_swift.wrapped = swift.swift_retry_iter(resp_half, image_size, self.store, loc.store_location, manager)
        self.assertEqual(5120, image_size)
        expected_data = b'*' * FIVE_KB
        data = b''
        for chunk in image_swift:
            data += chunk
        self.assertEqual(expected_data, data)

    def test_get_with_http_auth(self):
        """
        Test a retrieval from Swift with an HTTP authurl. This is
        specified either via a Location header with swift+http:// or using
        http:// in the swift_store_auth_address config value
        """
        loc = location.get_location_from_uri_and_backend('swift+http://%s:key@auth_address/glance/%s' % (self.swift_store_user, FAKE_UUID), 'swift1', conf=self.conf)
        ctxt = mock.MagicMock()
        image_swift, image_size = self.store.get(loc, context=ctxt)
        self.assertEqual(5120, image_size)
        expected_data = b'*' * FIVE_KB
        data = b''
        for chunk in image_swift:
            data += chunk
        self.assertEqual(expected_data, data)

    def test_get_non_existing(self):
        """
        Test that trying to retrieve a swift that doesn't exist
        raises an error
        """
        loc = location.get_location_from_uri_and_backend('swift://%s:key@authurl/glance/noexist' % self.swift_store_user, 'swift1', conf=self.conf)
        self.assertRaises(exceptions.NotFound, self.store.get, loc)

    def test_buffered_reader_opts(self):
        self.config(group='swift1', swift_buffer_on_upload=True)
        self.config(group='swift1', swift_upload_buffer_dir=self.test_dir)
        try:
            self.store = Store(self.conf, backend='swift1')
        except exceptions.BadStoreConfiguration:
            self.fail('Buffered Reader exception raised when it should not have been')

    def test_buffered_reader_with_invalid_path(self):
        self.config(group='swift1', swift_buffer_on_upload=True)
        self.config(group='swift1', swift_upload_buffer_dir='/some/path')
        self.store = Store(self.conf, backend='swift1')
        self.assertRaises(exceptions.BadStoreConfiguration, self.store.configure)

    def test_buffered_reader_with_no_path_given(self):
        self.config(group='swift1', swift_buffer_on_upload=True)
        self.store = Store(self.conf, backend='swift1')
        self.assertRaises(exceptions.BadStoreConfiguration, self.store.configure)

    @mock.patch('glance_store._drivers.swift.utils.is_multiple_swift_store_accounts_enabled', mock.Mock(return_value=False))
    def test_add(self):
        """Test that we can add an image via the swift backend."""
        importlib.reload(swift)
        self.mock_keystone_client()
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        expected_swift_size = FIVE_KB
        expected_swift_contents = b'*' * expected_swift_size
        expected_checksum = md5(expected_swift_contents, usedforsecurity=False).hexdigest()
        expected_image_id = str(uuid.uuid4())
        loc = 'swift+https://tenant%%3Auser1:key@localhost:8080/glance/%s'
        expected_location = loc % expected_image_id
        image_swift = io.BytesIO(expected_swift_contents)
        global SWIFT_PUT_OBJECT_CALLS
        SWIFT_PUT_OBJECT_CALLS = 0
        loc, size, checksum, metadata = self.store.add(expected_image_id, image_swift, expected_swift_size)
        self.assertEqual('swift1', metadata['store'])
        self.assertEqual(expected_location, loc)
        self.assertEqual(expected_swift_size, size)
        self.assertEqual(expected_checksum, checksum)
        self.assertEqual(1, SWIFT_PUT_OBJECT_CALLS)
        loc = location.get_location_from_uri_and_backend(expected_location, 'swift1', conf=self.conf)
        new_image_swift, new_image_size = self.store.get(loc)
        new_image_contents = b''.join([chunk for chunk in new_image_swift])
        new_image_swift_size = len(new_image_swift)
        self.assertEqual(expected_swift_contents, new_image_contents)
        self.assertEqual(expected_swift_size, new_image_swift_size)

    def test_add_multi_store(self):
        conf = copy.deepcopy(SWIFT_CONF)
        conf['default_swift_reference'] = 'store_2'
        self.config(group='swift1', **conf)
        importlib.reload(swift)
        self.mock_keystone_client()
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        expected_swift_size = FIVE_KB
        expected_swift_contents = b'*' * expected_swift_size
        expected_image_id = str(uuid.uuid4())
        image_swift = io.BytesIO(expected_swift_contents)
        global SWIFT_PUT_OBJECT_CALLS
        SWIFT_PUT_OBJECT_CALLS = 0
        loc = 'swift+config://store_2/glance/%s'
        expected_location = loc % expected_image_id
        location, size, checksum, arg = self.store.add(expected_image_id, image_swift, expected_swift_size)
        self.assertEqual('swift1', arg['store'])
        self.assertEqual(expected_location, location)

    @mock.patch('glance_store._drivers.swift.utils.is_multiple_swift_store_accounts_enabled', mock.Mock(return_value=False))
    def test_multi_tenant_image_add_uses_users_context(self):
        expected_swift_size = FIVE_KB
        expected_swift_contents = b'*' * expected_swift_size
        expected_image_id = str(uuid.uuid4())
        expected_container = 'container_' + expected_image_id
        loc = 'swift+https://some_endpoint/%s/%s'
        expected_location = loc % (expected_container, expected_image_id)
        image_swift = io.BytesIO(expected_swift_contents)
        global SWIFT_PUT_OBJECT_CALLS
        SWIFT_PUT_OBJECT_CALLS = 0
        self.config(group='swift1', swift_store_container='container')
        self.config(group='swift1', swift_store_create_container_on_put=True)
        self.config(group='swift1', swift_store_multi_tenant=True)
        service_catalog = [{'endpoint_links': [], 'endpoints': [{'adminURL': 'https://some_admin_endpoint', 'region': 'RegionOne', 'internalURL': 'https://some_internal_endpoint', 'publicURL': 'https://some_endpoint'}], 'type': 'object-store', 'name': 'Object Storage Service'}]
        ctxt = mock.MagicMock(user='user', tenant='tenant', auth_token='123', service_catalog=service_catalog)
        store = swift.MultiTenantStore(self.conf, backend='swift1')
        store.configure()
        loc, size, checksum, metadata = store.add(expected_image_id, image_swift, expected_swift_size, context=ctxt)
        self.assertEqual('swift1', metadata['store'])
        self.assertEqual(expected_location, loc)

    @mock.patch('glance_store._drivers.swift.utils.is_multiple_swift_store_accounts_enabled', mock.Mock(return_value=True))
    def test_add_auth_url_variations(self):
        """
        Test that we can add an image via the swift backend with
        a variety of different auth_address values
        """
        conf = copy.deepcopy(SWIFT_CONF)
        self.config(group='swift1', **conf)
        variations = {'store_4': 'swift+config://store_4/glance/%s', 'store_5': 'swift+config://store_5/glance/%s', 'store_6': 'swift+config://store_6/glance/%s'}
        for variation, expected_location in variations.items():
            image_id = str(uuid.uuid4())
            expected_location = expected_location % image_id
            expected_swift_size = FIVE_KB
            expected_swift_contents = b'*' * expected_swift_size
            expected_checksum = md5(expected_swift_contents, usedforsecurity=False).hexdigest()
            image_swift = io.BytesIO(expected_swift_contents)
            global SWIFT_PUT_OBJECT_CALLS
            SWIFT_PUT_OBJECT_CALLS = 0
            conf['default_swift_reference'] = variation
            self.config(group='swift1', **conf)
            importlib.reload(swift)
            self.mock_keystone_client()
            self.store = Store(self.conf, backend='swift1')
            self.store.configure()
            loc, size, checksum, metadata = self.store.add(image_id, image_swift, expected_swift_size)
            self.assertEqual('swift1', metadata['store'])
            self.assertEqual(expected_location, loc)
            self.assertEqual(expected_swift_size, size)
            self.assertEqual(expected_checksum, checksum)
            self.assertEqual(1, SWIFT_PUT_OBJECT_CALLS)
            loc = location.get_location_from_uri_and_backend(expected_location, 'swift1', conf=self.conf)
            new_image_swift, new_image_size = self.store.get(loc)
            new_image_contents = b''.join([chunk for chunk in new_image_swift])
            new_image_swift_size = len(new_image_swift)
            self.assertEqual(expected_swift_contents, new_image_contents)
            self.assertEqual(expected_swift_size, new_image_swift_size)

    def test_add_no_container_no_create(self):
        """
        Tests that adding an image with a non-existing container
        raises an appropriate exception
        """
        conf = copy.deepcopy(SWIFT_CONF)
        conf['swift_store_user'] = 'tenant:user'
        conf['swift_store_create_container_on_put'] = False
        conf['swift_store_container'] = 'noexist'
        self.config(group='swift1', **conf)
        importlib.reload(swift)
        self.mock_keystone_client()
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        image_swift = io.BytesIO(b'nevergonnamakeit')
        global SWIFT_PUT_OBJECT_CALLS
        SWIFT_PUT_OBJECT_CALLS = 0
        exception_caught = False
        try:
            self.store.add(str(uuid.uuid4()), image_swift, 0)
        except exceptions.BackendException as e:
            exception_caught = True
            self.assertIn('container noexist does not exist in Swift', encodeutils.exception_to_unicode(e))
        self.assertTrue(exception_caught)
        self.assertEqual(0, SWIFT_PUT_OBJECT_CALLS)

    @mock.patch('glance_store._drivers.swift.utils.is_multiple_swift_store_accounts_enabled', mock.Mock(return_value=True))
    def test_add_no_container_and_create(self):
        """
        Tests that adding an image with a non-existing container
        creates the container automatically if flag is set
        """
        expected_swift_size = FIVE_KB
        expected_swift_contents = b'*' * expected_swift_size
        expected_checksum = md5(expected_swift_contents, usedforsecurity=False).hexdigest()
        expected_image_id = str(uuid.uuid4())
        loc = 'swift+config://ref1/noexist/%s'
        expected_location = loc % expected_image_id
        image_swift = io.BytesIO(expected_swift_contents)
        global SWIFT_PUT_OBJECT_CALLS
        SWIFT_PUT_OBJECT_CALLS = 0
        conf = copy.deepcopy(SWIFT_CONF)
        conf['swift_store_user'] = 'tenant:user'
        conf['swift_store_create_container_on_put'] = True
        conf['swift_store_container'] = 'noexist'
        self.config(group='swift1', **conf)
        importlib.reload(swift)
        self.mock_keystone_client()
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        loc, size, checksum, metadata = self.store.add(expected_image_id, image_swift, expected_swift_size)
        self.assertEqual('swift1', metadata['store'])
        self.assertEqual(expected_location, loc)
        self.assertEqual(expected_swift_size, size)
        self.assertEqual(expected_checksum, checksum)
        self.assertEqual(1, SWIFT_PUT_OBJECT_CALLS)
        loc = location.get_location_from_uri_and_backend(expected_location, 'swift1', conf=self.conf)
        new_image_swift, new_image_size = self.store.get(loc)
        new_image_contents = b''.join([chunk for chunk in new_image_swift])
        new_image_swift_size = len(new_image_swift)
        self.assertEqual(expected_swift_contents, new_image_contents)
        self.assertEqual(expected_swift_size, new_image_swift_size)

    @mock.patch('glance_store._drivers.swift.utils.is_multiple_swift_store_accounts_enabled', mock.Mock(return_value=True))
    def test_add_no_container_and_multiple_containers_create(self):
        """
        Tests that adding an image with a non-existing container while using
        multi containers will create the container automatically if flag is set
        """
        expected_swift_size = FIVE_KB
        expected_swift_contents = b'*' * expected_swift_size
        expected_checksum = md5(expected_swift_contents, usedforsecurity=False).hexdigest()
        expected_image_id = str(uuid.uuid4())
        container = 'randomname_' + expected_image_id[:2]
        loc = 'swift+config://ref1/%s/%s'
        expected_location = loc % (container, expected_image_id)
        image_swift = io.BytesIO(expected_swift_contents)
        global SWIFT_PUT_OBJECT_CALLS
        SWIFT_PUT_OBJECT_CALLS = 0
        conf = copy.deepcopy(SWIFT_CONF)
        conf['swift_store_user'] = 'tenant:user'
        conf['swift_store_create_container_on_put'] = True
        conf['swift_store_container'] = 'randomname'
        conf['swift_store_multiple_containers_seed'] = 2
        self.config(group='swift1', **conf)
        importlib.reload(swift)
        self.mock_keystone_client()
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        loc, size, checksum, metadata = self.store.add(expected_image_id, image_swift, expected_swift_size)
        self.assertEqual('swift1', metadata['store'])
        self.assertEqual(expected_location, loc)
        self.assertEqual(expected_swift_size, size)
        self.assertEqual(expected_checksum, checksum)
        self.assertEqual(1, SWIFT_PUT_OBJECT_CALLS)
        loc = location.get_location_from_uri_and_backend(expected_location, 'swift1', conf=self.conf)
        new_image_swift, new_image_size = self.store.get(loc)
        new_image_contents = b''.join([chunk for chunk in new_image_swift])
        new_image_swift_size = len(new_image_swift)
        self.assertEqual(expected_swift_contents, new_image_contents)
        self.assertEqual(expected_swift_size, new_image_swift_size)

    @mock.patch('glance_store._drivers.swift.utils.is_multiple_swift_store_accounts_enabled', mock.Mock(return_value=True))
    def test_add_no_container_and_multiple_containers_no_create(self):
        """
        Tests that adding an image with a non-existing container while using
        multiple containers raises an appropriate exception
        """
        conf = copy.deepcopy(SWIFT_CONF)
        conf['swift_store_user'] = 'tenant:user'
        conf['swift_store_create_container_on_put'] = False
        conf['swift_store_container'] = 'randomname'
        conf['swift_store_multiple_containers_seed'] = 2
        self.config(group='swift1', **conf)
        importlib.reload(swift)
        self.mock_keystone_client()
        expected_image_id = str(uuid.uuid4())
        expected_container = 'randomname_' + expected_image_id[:2]
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        image_swift = io.BytesIO(b'nevergonnamakeit')
        global SWIFT_PUT_OBJECT_CALLS
        SWIFT_PUT_OBJECT_CALLS = 0
        exception_caught = False
        try:
            self.store.add(expected_image_id, image_swift, 0)
        except exceptions.BackendException as e:
            exception_caught = True
            expected_msg = 'container %s does not exist in Swift'
            expected_msg = expected_msg % expected_container
            self.assertIn(expected_msg, encodeutils.exception_to_unicode(e))
        self.assertTrue(exception_caught)
        self.assertEqual(0, SWIFT_PUT_OBJECT_CALLS)

    @mock.patch('glance_store._drivers.swift.utils.is_multiple_swift_store_accounts_enabled', mock.Mock(return_value=True))
    def test_add_with_verifier(self):
        """Test that the verifier is updated when verifier is provided."""
        swift_size = FIVE_KB
        base_byte = b'12345678'
        swift_contents = base_byte * (swift_size // 8)
        image_id = str(uuid.uuid4())
        image_swift = io.BytesIO(swift_contents)
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        orig_max_size = self.store.large_object_size
        orig_temp_size = self.store.large_object_chunk_size
        custom_size = units.Ki
        verifier = mock.MagicMock(name='mock_verifier')
        try:
            self.store.large_object_size = custom_size
            self.store.large_object_chunk_size = custom_size
            self.store.add(image_id, image_swift, swift_size, verifier=verifier)
        finally:
            self.store.large_object_chunk_size = orig_temp_size
            self.store.large_object_size = orig_max_size
        self.assertEqual(2 * swift_size / custom_size, verifier.update.call_count)
        swift_contents_piece = base_byte * (custom_size // 8)
        calls = [mock.call(swift_contents_piece), mock.call(b''), mock.call(swift_contents_piece), mock.call(b''), mock.call(swift_contents_piece), mock.call(b''), mock.call(swift_contents_piece), mock.call(b''), mock.call(swift_contents_piece), mock.call(b'')]
        verifier.update.assert_has_calls(calls)

    @mock.patch('glance_store._drivers.swift.utils.is_multiple_swift_store_accounts_enabled', mock.Mock(return_value=True))
    def test_add_with_verifier_small(self):
        """Test that the verifier is updated for smaller images."""
        swift_size = FIVE_KB
        base_byte = b'12345678'
        swift_contents = base_byte * (swift_size // 8)
        image_id = str(uuid.uuid4())
        image_swift = io.BytesIO(swift_contents)
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        orig_max_size = self.store.large_object_size
        orig_temp_size = self.store.large_object_chunk_size
        custom_size = 6 * units.Ki
        verifier = mock.MagicMock(name='mock_verifier')
        try:
            self.store.large_object_size = custom_size
            self.store.large_object_chunk_size = custom_size
            self.store.add(image_id, image_swift, swift_size, verifier=verifier)
        finally:
            self.store.large_object_chunk_size = orig_temp_size
            self.store.large_object_size = orig_max_size
        self.assertEqual(2, verifier.update.call_count)
        swift_contents_piece = base_byte * (swift_size // 8)
        calls = [mock.call(swift_contents_piece), mock.call(b'')]
        verifier.update.assert_has_calls(calls)

    @mock.patch('glance_store._drivers.swift.utils.is_multiple_swift_store_accounts_enabled', mock.Mock(return_value=False))
    def test_multi_container_doesnt_impact_multi_tenant_add(self):
        expected_swift_size = FIVE_KB
        expected_swift_contents = b'*' * expected_swift_size
        expected_image_id = str(uuid.uuid4())
        expected_container = 'container_' + expected_image_id
        loc = 'swift+https://some_endpoint/%s/%s'
        expected_location = loc % (expected_container, expected_image_id)
        image_swift = io.BytesIO(expected_swift_contents)
        global SWIFT_PUT_OBJECT_CALLS
        SWIFT_PUT_OBJECT_CALLS = 0
        self.config(group='swift1', swift_store_container='container')
        self.config(group='swift1', swift_store_create_container_on_put=True)
        self.config(group='swift1', swift_store_multiple_containers_seed=2)
        service_catalog = [{'endpoint_links': [], 'endpoints': [{'adminURL': 'https://some_admin_endpoint', 'region': 'RegionOne', 'internalURL': 'https://some_internal_endpoint', 'publicURL': 'https://some_endpoint'}], 'type': 'object-store', 'name': 'Object Storage Service'}]
        ctxt = mock.MagicMock(user='user', tenant='tenant', auth_token='123', service_catalog=service_catalog)
        store = swift.MultiTenantStore(self.conf, backend='swift1')
        store.configure()
        location, size, checksum, metadata = store.add(expected_image_id, image_swift, expected_swift_size, context=ctxt)
        self.assertEqual('swift1', metadata['store'])
        self.assertEqual(expected_location, location)

    @mock.patch('glance_store._drivers.swift.utils.is_multiple_swift_store_accounts_enabled', mock.Mock(return_value=True))
    def test_add_large_object(self):
        """
        Tests that adding a very large image. We simulate the large
        object by setting store.large_object_size to a small number
        and then verify that there have been a number of calls to
        put_object()...
        """
        expected_swift_size = FIVE_KB
        expected_swift_contents = b'*' * expected_swift_size
        expected_checksum = md5(expected_swift_contents, usedforsecurity=False).hexdigest()
        expected_image_id = str(uuid.uuid4())
        loc = 'swift+config://ref1/glance/%s'
        expected_location = loc % expected_image_id
        image_swift = io.BytesIO(expected_swift_contents)
        global SWIFT_PUT_OBJECT_CALLS
        SWIFT_PUT_OBJECT_CALLS = 0
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        orig_max_size = self.store.large_object_size
        orig_temp_size = self.store.large_object_chunk_size
        try:
            self.store.large_object_size = units.Ki
            self.store.large_object_chunk_size = units.Ki
            loc, size, checksum, metadata = self.store.add(expected_image_id, image_swift, expected_swift_size)
        finally:
            self.store.large_object_chunk_size = orig_temp_size
            self.store.large_object_size = orig_max_size
        self.assertEqual('swift1', metadata['store'])
        self.assertEqual(expected_location, loc)
        self.assertEqual(expected_swift_size, size)
        self.assertEqual(expected_checksum, checksum)
        self.assertEqual(6, SWIFT_PUT_OBJECT_CALLS)
        loc = location.get_location_from_uri_and_backend(expected_location, 'swift1', conf=self.conf)
        new_image_swift, new_image_size = self.store.get(loc)
        new_image_contents = b''.join([chunk for chunk in new_image_swift])
        new_image_swift_size = len(new_image_contents)
        self.assertEqual(expected_swift_contents, new_image_contents)
        self.assertEqual(expected_swift_size, new_image_swift_size)

    def test_add_large_object_zero_size(self):
        """
        Tests that adding an image to Swift which has both an unknown size and
        exceeds Swift's maximum limit of 5GB is correctly uploaded.

        We avoid the overhead of creating a 5GB object for this test by
        temporarily setting MAX_SWIFT_OBJECT_SIZE to 1KB, and then adding
        an object of 5KB.

        Bug lp:891738
        """
        expected_swift_size = FIVE_KB
        expected_swift_contents = b'*' * expected_swift_size
        expected_checksum = md5(expected_swift_contents, usedforsecurity=False).hexdigest()
        expected_image_id = str(uuid.uuid4())
        loc = 'swift+config://ref1/glance/%s'
        expected_location = loc % expected_image_id
        image_swift = io.BytesIO(expected_swift_contents)
        global SWIFT_PUT_OBJECT_CALLS
        SWIFT_PUT_OBJECT_CALLS = 0
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        orig_max_size = self.store.large_object_size
        orig_temp_size = self.store.large_object_chunk_size
        global MAX_SWIFT_OBJECT_SIZE
        orig_max_swift_object_size = MAX_SWIFT_OBJECT_SIZE
        try:
            MAX_SWIFT_OBJECT_SIZE = units.Ki
            self.store.large_object_size = units.Ki
            self.store.large_object_chunk_size = units.Ki
            loc, size, checksum, metadata = self.store.add(expected_image_id, image_swift, 0)
        finally:
            self.store.large_object_chunk_size = orig_temp_size
            self.store.large_object_size = orig_max_size
            MAX_SWIFT_OBJECT_SIZE = orig_max_swift_object_size
        self.assertEqual('swift1', metadata['store'])
        self.assertEqual(expected_location, loc)
        self.assertEqual(expected_swift_size, size)
        self.assertEqual(expected_checksum, checksum)
        self.assertEqual(6, SWIFT_PUT_OBJECT_CALLS)
        loc = location.get_location_from_uri_and_backend(expected_location, 'swift1', conf=self.conf)
        new_image_swift, new_image_size = self.store.get(loc)
        new_image_contents = b''.join([chunk for chunk in new_image_swift])
        new_image_swift_size = len(new_image_contents)
        self.assertEqual(expected_swift_contents, new_image_contents)
        self.assertEqual(expected_swift_size, new_image_swift_size)

    def test_location_url_prefix_is_set(self):
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        expected_url_prefix = 'swift+config://ref1/glance/'
        self.assertEqual(expected_url_prefix, self.store.url_prefix)

    def test_add_already_existing(self):
        """
        Tests that adding an image with an existing identifier
        raises an appropriate exception
        """
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        image_swift = io.BytesIO(b'nevergonnamakeit')
        self.assertRaises(exceptions.Duplicate, self.store.add, FAKE_UUID, image_swift, 0)

    def _option_required(self, key):
        conf = self.getConfig()
        conf[key] = None
        try:
            self.config(group='swift1', **conf)
            self.store = Store(self.conf, backend='swift1')
            return not self.store.is_capable(capabilities.BitMasks.WRITE_ACCESS)
        except Exception:
            return False

    def test_no_store_credentials(self):
        """
        Tests that options without a valid credentials disables the add method
        """
        self.store = Store(self.conf, backend='swift1')
        self.store.ref_params = {'ref1': {'auth_address': 'authurl.com', 'user': '', 'key': ''}}
        self.store.configure()
        self.assertFalse(self.store.is_capable(capabilities.BitMasks.WRITE_ACCESS))

    def test_no_auth_address(self):
        """
        Tests that options without auth address disables the add method
        """
        self.store = Store(self.conf, backend='swift1')
        self.store.ref_params = {'ref1': {'auth_address': '', 'user': 'user1', 'key': 'key1'}}
        self.store.configure()
        self.assertFalse(self.store.is_capable(capabilities.BitMasks.WRITE_ACCESS))

    def test_delete(self):
        """
        Test we can delete an existing image in the swift store
        """
        conf = copy.deepcopy(SWIFT_CONF)
        self.config(group='swift1', **conf)
        importlib.reload(swift)
        self.mock_keystone_client()
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        uri = 'swift://%s:key@authurl/glance/%s' % (self.swift_store_user, FAKE_UUID)
        loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
        self.store.delete(loc)
        self.assertRaises(exceptions.NotFound, self.store.get, loc)

    @mock.patch.object(swiftclient.client, 'delete_object')
    def test_delete_slo(self, mock_del_obj):
        """
        Test we can delete an existing image stored as SLO, static large object
        """
        conf = copy.deepcopy(SWIFT_CONF)
        self.config(group='swift1', **conf)
        importlib.reload(swift)
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        uri = 'swift://%s:key@authurl/glance/%s' % (self.swift_store_user, FAKE_UUID2)
        loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
        self.store.delete(loc)
        self.assertEqual(1, mock_del_obj.call_count)
        _, kwargs = mock_del_obj.call_args
        self.assertEqual('multipart-manifest=delete', kwargs.get('query_string'))

    @mock.patch.object(swiftclient.client, 'delete_object')
    def test_delete_nonslo_not_deleted_as_slo(self, mock_del_obj):
        """
        Test that non-SLOs are not being deleted the SLO way
        """
        conf = copy.deepcopy(SWIFT_CONF)
        self.config(group='swift1', **conf)
        importlib.reload(swift)
        self.mock_keystone_client()
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        uri = 'swift://%s:key@authurl/glance/%s' % (self.swift_store_user, FAKE_UUID)
        loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
        self.store.delete(loc)
        self.assertEqual(1, mock_del_obj.call_count)
        _, kwargs = mock_del_obj.call_args
        self.assertIsNone(kwargs.get('query_string'))

    def test_delete_with_reference_params(self):
        """
        Test we can delete an existing image in the swift store
        """
        conf = copy.deepcopy(SWIFT_CONF)
        self.config(group='swift1', **conf)
        importlib.reload(swift)
        self.mock_keystone_client()
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        uri = 'swift+config://ref1/glance/%s' % FAKE_UUID
        loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
        self.store.delete(loc)
        self.assertRaises(exceptions.NotFound, self.store.get, loc)

    def test_delete_non_existing(self):
        """
        Test that trying to delete a swift that doesn't exist
        raises an error
        """
        conf = copy.deepcopy(SWIFT_CONF)
        self.config(group='swift1', **conf)
        importlib.reload(swift)
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        loc = location.get_location_from_uri_and_backend('swift://%s:key@authurl/glance/noexist' % self.swift_store_user, 'swift1', conf=self.conf)
        self.assertRaises(exceptions.NotFound, self.store.delete, loc)

    def test_delete_with_some_segments_failing(self):
        """
        Tests that delete of a segmented object recovers from error(s) while
        deleting one or more segments.
        To test this we add a segmented object first and then delete it, while
        simulating errors on one or more segments.
        """
        test_image_id = str(uuid.uuid4())

        def fake_head_object(container, object_name):
            object_manifest = '/'.join([container, object_name]) + '-'
            return {'x-object-manifest': object_manifest}

        def fake_get_container(container, **kwargs):
            return (None, [{'name': '%s-%03d' % (test_image_id, x)} for x in range(1, 6)])

        def fake_delete_object(container, object_name):
            global SWIFT_DELETE_OBJECT_CALLS
            SWIFT_DELETE_OBJECT_CALLS += 1
            if object_name.endswith('-001') or object_name.endswith('-003'):
                raise swiftclient.ClientException('Object DELETE failed')
            else:
                pass
        conf = copy.deepcopy(SWIFT_CONF)
        self.config(group='swift1', **conf)
        importlib.reload(swift)
        self.store = Store(self.conf, backend='swift1')
        self.store.configure()
        loc_uri = 'swift+https://%s:key@localhost:8080/glance/%s'
        loc_uri = loc_uri % (self.swift_store_user, test_image_id)
        loc = location.get_location_from_uri_and_backend(loc_uri, 'swift1', conf=self.conf)
        conn = self.store.get_connection(loc.store_location)
        conn.delete_object = fake_delete_object
        conn.head_object = fake_head_object
        conn.get_container = fake_get_container
        global SWIFT_DELETE_OBJECT_CALLS
        SWIFT_DELETE_OBJECT_CALLS = 0
        self.store.delete(loc, connection=conn)
        self.assertEqual(6, SWIFT_DELETE_OBJECT_CALLS)

    def test_read_acl_public(self):
        """
        Test that we can set a public read acl.
        """
        self.config(group='swift1', swift_store_config_file=None)
        self.config(group='swift1', swift_store_multi_tenant=True)
        store = Store(self.conf, backend='swift1')
        store.configure()
        uri = 'swift+http://storeurl/glance/%s' % FAKE_UUID
        loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
        ctxt = mock.MagicMock()
        store.set_acls(loc, public=True, context=ctxt)
        container_headers = swiftclient.client.head_container('x', 'y', 'glance')
        self.assertEqual('*:*', container_headers['X-Container-Read'])

    def test_read_acl_tenants(self):
        """
        Test that we can set read acl for tenants.
        """
        self.config(group='swift1', swift_store_config_file=None)
        self.config(group='swift1', swift_store_multi_tenant=True)
        store = Store(self.conf, backend='swift1')
        store.configure()
        uri = 'swift+http://storeurl/glance/%s' % FAKE_UUID
        loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
        read_tenants = ['matt', 'mark']
        ctxt = mock.MagicMock()
        store.set_acls(loc, read_tenants=read_tenants, context=ctxt)
        container_headers = swiftclient.client.head_container('x', 'y', 'glance')
        self.assertEqual('matt:*,mark:*', container_headers['X-Container-Read'])

    def test_write_acls(self):
        """
        Test that we can set write acl for tenants.
        """
        self.config(group='swift1', swift_store_config_file=None)
        self.config(group='swift1', swift_store_multi_tenant=True)
        store = Store(self.conf, backend='swift1')
        store.configure()
        uri = 'swift+http://storeurl/glance/%s' % FAKE_UUID
        loc = location.get_location_from_uri_and_backend(uri, 'swift1', conf=self.conf)
        read_tenants = ['frank', 'jim']
        ctxt = mock.MagicMock()
        store.set_acls(loc, write_tenants=read_tenants, context=ctxt)
        container_headers = swiftclient.client.head_container('x', 'y', 'glance')
        self.assertEqual('frank:*,jim:*', container_headers['X-Container-Write'])

    @mock.patch('glance_store._drivers.swift.connection_manager.MultiTenantConnectionManager')
    def test_get_connection_manager_multi_tenant(self, manager_class):
        manager = mock.MagicMock()
        manager_class.return_value = manager
        self.config(group='swift1', swift_store_config_file=None)
        self.config(group='swift1', swift_store_multi_tenant=True)
        store = Store(self.conf, backend='swift1')
        store.configure()
        loc = mock.MagicMock()
        self.assertEqual(store.get_manager(loc), manager)

    @mock.patch('glance_store._drivers.swift.connection_manager.SingleTenantConnectionManager')
    def test_get_connection_manager_single_tenant(self, manager_class):
        manager = mock.MagicMock()
        manager_class.return_value = manager
        store = Store(self.conf, backend='swift1')
        store.configure()
        loc = mock.MagicMock()
        self.assertEqual(store.get_manager(loc), manager)

    def test_get_connection_manager_failed(self):
        store = swift.BaseStore(mock.MagicMock())
        loc = mock.MagicMock()
        self.assertRaises(NotImplementedError, store.get_manager, loc)

    def test_init_client_multi_tenant(self):
        """Test that keystone client was initialized correctly"""
        with mock.patch.object(swift.MultiTenantStore, '_set_url_prefix'):
            self._init_client(verify=True, swift_store_multi_tenant=True, swift_store_config_file=None)

    def test_init_client_multi_tenant_swift_cacert(self):
        """Test that keystone client was initialized with swift cacert"""
        with mock.patch.object(swift.MultiTenantStore, '_set_url_prefix'):
            self._init_client(verify='/foo/bar', swift_store_multi_tenant=True, swift_store_config_file=None, swift_store_cacert='/foo/bar')

    def test_init_client_multi_tenant_insecure(self):
        """
        Test that keystone client was initialized correctly with no
        certificate verification.
        """
        with mock.patch.object(swift.MultiTenantStore, '_set_url_prefix'):
            self._init_client(verify=False, swift_store_multi_tenant=True, swift_store_auth_insecure=True, swift_store_config_file=None)

    @mock.patch('glance_store._drivers.swift.store.ks_identity')
    @mock.patch('glance_store._drivers.swift.store.ks_session')
    @mock.patch('glance_store._drivers.swift.store.ks_client')
    def _init_client(self, mock_client, mock_session, mock_identity, verify, **kwargs):
        self.config(group='swift1', **kwargs)
        store = Store(self.conf, backend='swift1')
        store.configure()
        ref_params = sutils.SwiftParams(self.conf, backend='swift1').params
        default_ref = getattr(self.conf, 'swift1').default_swift_reference
        default_swift_reference = ref_params.get(default_ref)
        trustee_session = mock.MagicMock()
        trustor_session = mock.MagicMock()
        main_session = mock.MagicMock()
        trustee_client = mock.MagicMock()
        trustee_client.session.get_user_id.return_value = 'fake_user'
        trustor_client = mock.MagicMock()
        trustor_client.session.auth.get_auth_ref.return_value = {'roles': [{'name': 'fake_role'}]}
        trustor_client.trusts.create.return_value = mock.MagicMock(id='fake_trust')
        main_client = mock.MagicMock()
        mock_session.Session.side_effect = [trustor_session, trustee_session, main_session]
        mock_client.Client.side_effect = [trustor_client, trustee_client, main_client]
        ctxt = mock.MagicMock()
        client = store.init_client(location=mock.MagicMock(), context=ctxt)
        mock_identity.V3Token.assert_called_once_with(auth_url=default_swift_reference.get('auth_address'), token=ctxt.auth_token, project_id=ctxt.project_id)
        mock_session.Session.assert_any_call(auth=mock_identity.V3Token(), verify=verify)
        mock_client.Client.assert_any_call(session=trustor_session)
        tenant_name, user = default_swift_reference.get('user').split(':')
        mock_identity.V3Password.assert_any_call(auth_url=default_swift_reference.get('auth_address'), username=user, password=default_swift_reference.get('key'), project_name=tenant_name, user_domain_id=default_swift_reference.get('user_domain_id'), user_domain_name=default_swift_reference.get('user_domain_name'), project_domain_id=default_swift_reference.get('project_domain_id'), project_domain_name=default_swift_reference.get('project_domain_name'))
        mock_session.Session.assert_any_call(auth=mock_identity.V3Password(), verify=verify)
        mock_client.Client.assert_any_call(session=trustee_session)
        trustor_client.trusts.create.assert_called_once_with(trustee_user='fake_user', trustor_user=ctxt.user_id, project=ctxt.project_id, impersonation=True, role_names=['fake_role'])
        mock_identity.V3Password.assert_any_call(auth_url=default_swift_reference.get('auth_address'), username=user, password=default_swift_reference.get('key'), trust_id='fake_trust', user_domain_id=default_swift_reference.get('user_domain_id'), user_domain_name=default_swift_reference.get('user_domain_name'), project_domain_id=default_swift_reference.get('project_domain_id'), project_domain_name=default_swift_reference.get('project_domain_name'))
        mock_client.Client.assert_any_call(session=main_session)
        self.assertEqual(main_client, client)
import datetime
from unittest import mock
import glance_store
from oslo_config import cfg
import oslo_messaging
import webob
import glance.async_
from glance.common import exception
from glance.common import timeutils
import glance.context
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
from glance.tests import utils
class TestImageNotifications(utils.BaseTestCase):
    """Test Image Notifications work"""

    def setUp(self):
        super(TestImageNotifications, self).setUp()
        self.image = ImageStub(image_id=UUID1, name='image-1', status='active', size=1024, created_at=DATETIME, updated_at=DATETIME, owner=TENANT1, visibility='public', container_format='ami', virtual_size=2048, tags=['one', 'two'], disk_format='ami', min_ram=128, min_disk=10, checksum='ca425b88f047ce8ec45ee90e813ada91', locations=['http://127.0.0.1'])
        self.context = glance.context.RequestContext(tenant=TENANT2, user=USER1)
        self.image_repo_stub = ImageRepoStub()
        self.notifier = unit_test_utils.FakeNotifier()
        self.image_repo_proxy = glance.notifier.ImageRepoProxy(self.image_repo_stub, self.context, self.notifier)
        self.image_proxy = glance.notifier.ImageProxy(self.image, self.context, self.notifier)

    def test_image_save_notification(self):
        self.image_repo_proxy.save(self.image_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('image.update', output_log['event_type'])
        self.assertEqual(self.image.image_id, output_log['payload']['id'])
        if 'location' in output_log['payload']:
            self.fail('Notification contained location field.')

    def test_image_save_notification_disabled(self):
        self.config(disabled_notifications=['image.update'])
        self.image_repo_proxy.save(self.image_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_image_add_notification(self):
        self.image_repo_proxy.add(self.image_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('image.create', output_log['event_type'])
        self.assertEqual(self.image.image_id, output_log['payload']['id'])
        if 'location' in output_log['payload']:
            self.fail('Notification contained location field.')

    def test_image_add_notification_disabled(self):
        self.config(disabled_notifications=['image.create'])
        self.image_repo_proxy.add(self.image_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_image_delete_notification(self):
        self.image_repo_proxy.remove(self.image_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('image.delete', output_log['event_type'])
        self.assertEqual(self.image.image_id, output_log['payload']['id'])
        self.assertTrue(output_log['payload']['deleted'])
        if 'location' in output_log['payload']:
            self.fail('Notification contained location field.')

    def test_image_delete_notification_disabled(self):
        self.config(disabled_notifications=['image.delete'])
        self.image_repo_proxy.remove(self.image_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_image_get(self):
        image = self.image_repo_proxy.get(UUID1)
        self.assertIsInstance(image, glance.notifier.ImageProxy)
        self.assertEqual('image_from_get', image.repo)

    def test_image_list(self):
        images = self.image_repo_proxy.list()
        self.assertIsInstance(images[0], glance.notifier.ImageProxy)
        self.assertEqual('images_from_list', images[0].repo)

    def test_image_get_data_should_call_next_image_get_data(self):
        with mock.patch.object(self.image, 'get_data') as get_data_mock:
            self.image_proxy.get_data()
            self.assertTrue(get_data_mock.called)

    def test_image_get_data_notification(self):
        self.image_proxy.size = 10
        data = ''.join(self.image_proxy.get_data())
        self.assertEqual('0123456789', data)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('image.send', output_log['event_type'])
        self.assertEqual(self.image.image_id, output_log['payload']['image_id'])
        self.assertEqual(TENANT2, output_log['payload']['receiver_tenant_id'])
        self.assertEqual(USER1, output_log['payload']['receiver_user_id'])
        self.assertEqual(10, output_log['payload']['bytes_sent'])
        self.assertEqual(TENANT1, output_log['payload']['owner_id'])

    def test_image_get_data_notification_disabled(self):
        self.config(disabled_notifications=['image.send'])
        self.image_proxy.size = 10
        data = ''.join(self.image_proxy.get_data())
        self.assertEqual('0123456789', data)
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_image_get_data_size_mismatch(self):
        self.image_proxy.size = 11
        list(self.image_proxy.get_data())
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('ERROR', output_log['notification_type'])
        self.assertEqual('image.send', output_log['event_type'])
        self.assertEqual(self.image.image_id, output_log['payload']['image_id'])

    def test_image_set_data_prepare_notification(self):
        insurance = {'called': False}

        def data_iterator():
            output_logs = self.notifier.get_logs()
            self.assertEqual(1, len(output_logs))
            output_log = output_logs[0]
            self.assertEqual('INFO', output_log['notification_type'])
            self.assertEqual('image.prepare', output_log['event_type'])
            self.assertEqual(self.image.image_id, output_log['payload']['id'])
            self.assertEqual(['store1', 'store2'], output_log['payload']['os_glance_importing_to_stores'])
            self.assertEqual([], output_log['payload']['os_glance_failed_import'])
            yield 'abcd'
            yield 'efgh'
            insurance['called'] = True
        self.image_proxy.extra_properties['os_glance_importing_to_stores'] = 'store1,store2'
        self.image_proxy.extra_properties['os_glance_failed_import'] = ''
        self.image_proxy.set_data(data_iterator(), 8)
        self.assertTrue(insurance['called'])

    def test_image_set_data_prepare_notification_disabled(self):
        insurance = {'called': False}

        def data_iterator():
            output_logs = self.notifier.get_logs()
            self.assertEqual(0, len(output_logs))
            yield 'abcd'
            yield 'efgh'
            insurance['called'] = True
        self.config(disabled_notifications=['image.prepare'])
        self.image_proxy.set_data(data_iterator(), 8)
        self.assertTrue(insurance['called'])

    def test_image_set_data_upload_and_activate_notification(self):
        image = ImageStub(image_id=UUID1, name='image-1', status='queued', created_at=DATETIME, updated_at=DATETIME, owner=TENANT1, visibility='public')
        context = glance.context.RequestContext(tenant=TENANT2, user=USER1)
        fake_notifier = unit_test_utils.FakeNotifier()
        image_proxy = glance.notifier.ImageProxy(image, context, fake_notifier)

        def data_iterator():
            fake_notifier.log = []
            yield 'abcde'
            yield 'fghij'
            image_proxy.extra_properties['os_glance_importing_to_stores'] = 'store2'
        image_proxy.extra_properties['os_glance_importing_to_stores'] = 'store1,store2'
        image_proxy.extra_properties['os_glance_failed_import'] = ''
        image_proxy.set_data(data_iterator(), 10)
        output_logs = fake_notifier.get_logs()
        self.assertEqual(2, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('image.upload', output_log['event_type'])
        self.assertEqual(self.image.image_id, output_log['payload']['id'])
        self.assertEqual(['store2'], output_log['payload']['os_glance_importing_to_stores'])
        self.assertEqual([], output_log['payload']['os_glance_failed_import'])
        output_log = output_logs[1]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('image.activate', output_log['event_type'])
        self.assertEqual(self.image.image_id, output_log['payload']['id'])

    def test_image_set_data_upload_and_not_activate_notification(self):
        insurance = {'called': False}

        def data_iterator():
            self.notifier.log = []
            yield 'abcde'
            yield 'fghij'
            self.image_proxy.extra_properties['os_glance_importing_to_stores'] = 'store2'
            insurance['called'] = True
        self.image_proxy.set_data(data_iterator(), 10)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('image.upload', output_log['event_type'])
        self.assertEqual(self.image.image_id, output_log['payload']['id'])
        self.assertTrue(insurance['called'])

    def test_image_set_data_upload_and_activate_notification_disabled(self):
        insurance = {'called': False}
        image = ImageStub(image_id=UUID1, name='image-1', status='queued', created_at=DATETIME, updated_at=DATETIME, owner=TENANT1, visibility='public')
        context = glance.context.RequestContext(tenant=TENANT2, user=USER1)
        fake_notifier = unit_test_utils.FakeNotifier()
        image_proxy = glance.notifier.ImageProxy(image, context, fake_notifier)

        def data_iterator():
            fake_notifier.log = []
            yield 'abcde'
            yield 'fghij'
            insurance['called'] = True
        self.config(disabled_notifications=['image.activate', 'image.upload'])
        image_proxy.set_data(data_iterator(), 10)
        self.assertTrue(insurance['called'])
        output_logs = fake_notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_image_set_data_storage_full(self):

        def data_iterator():
            self.notifier.log = []
            yield 'abcde'
            raise glance_store.StorageFull(message='Modern Major General')
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, self.image_proxy.set_data, data_iterator(), 10)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('ERROR', output_log['notification_type'])
        self.assertEqual('image.upload', output_log['event_type'])
        self.assertIn('Modern Major General', output_log['payload'])

    def test_image_set_data_value_error(self):

        def data_iterator():
            self.notifier.log = []
            yield 'abcde'
            raise ValueError('value wrong')
        self.assertRaises(webob.exc.HTTPBadRequest, self.image_proxy.set_data, data_iterator(), 10)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('ERROR', output_log['notification_type'])
        self.assertEqual('image.upload', output_log['event_type'])
        self.assertIn('value wrong', output_log['payload'])

    def test_image_set_data_duplicate(self):

        def data_iterator():
            self.notifier.log = []
            yield 'abcde'
            raise exception.Duplicate('Cant have duplicates')
        self.assertRaises(webob.exc.HTTPConflict, self.image_proxy.set_data, data_iterator(), 10)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('ERROR', output_log['notification_type'])
        self.assertEqual('image.upload', output_log['event_type'])
        self.assertIn('Cant have duplicates', output_log['payload'])

    def test_image_set_data_storage_write_denied(self):

        def data_iterator():
            self.notifier.log = []
            yield 'abcde'
            raise glance_store.StorageWriteDenied(message='The Very Model')
        self.assertRaises(webob.exc.HTTPServiceUnavailable, self.image_proxy.set_data, data_iterator(), 10)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('ERROR', output_log['notification_type'])
        self.assertEqual('image.upload', output_log['event_type'])
        self.assertIn('The Very Model', output_log['payload'])

    def test_image_set_data_forbidden(self):

        def data_iterator():
            self.notifier.log = []
            yield 'abcde'
            raise exception.Forbidden('Not allowed')
        self.assertRaises(webob.exc.HTTPForbidden, self.image_proxy.set_data, data_iterator(), 10)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('ERROR', output_log['notification_type'])
        self.assertEqual('image.upload', output_log['event_type'])
        self.assertIn('Not allowed', output_log['payload'])

    def test_image_set_data_not_found(self):

        def data_iterator():
            self.notifier.log = []
            yield 'abcde'
            raise exception.NotFound('Not found')
        self.assertRaises(webob.exc.HTTPNotFound, self.image_proxy.set_data, data_iterator(), 10)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('ERROR', output_log['notification_type'])
        self.assertEqual('image.upload', output_log['event_type'])
        self.assertIn('Not found', output_log['payload'])

    def test_image_set_data_HTTP_error(self):

        def data_iterator():
            self.notifier.log = []
            yield 'abcde'
            raise webob.exc.HTTPError('Http issue')
        self.assertRaises(webob.exc.HTTPError, self.image_proxy.set_data, data_iterator(), 10)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('ERROR', output_log['notification_type'])
        self.assertEqual('image.upload', output_log['event_type'])
        self.assertIn('Http issue', output_log['payload'])

    def test_image_set_data_error(self):

        def data_iterator():
            self.notifier.log = []
            yield 'abcde'
            raise exception.GlanceException('Failed')
        self.assertRaises(exception.GlanceException, self.image_proxy.set_data, data_iterator(), 10)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('ERROR', output_log['notification_type'])
        self.assertEqual('image.upload', output_log['event_type'])
        self.assertIn('Failed', output_log['payload'])
from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store
from unittest import mock
from glance.common import exception
import glance.location
from glance.tests.unit import base as unit_test_base
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils
class TestStoreImage(utils.BaseTestCase):

    def setUp(self):
        locations = [{'url': '%s/%s' % (BASE_URI, UUID1), 'metadata': {}, 'status': 'active'}]
        self.image_stub = ImageStub(UUID1, 'active', locations)
        self.store_api = unit_test_utils.FakeStoreAPI()
        self.store_utils = unit_test_utils.FakeStoreUtils(self.store_api)
        super(TestStoreImage, self).setUp()

    def test_image_delete(self):
        image = glance.location.ImageProxy(self.image_stub, {}, self.store_api, self.store_utils)
        location = image.locations[0]
        self.assertEqual('active', image.status)
        self.store_api.get_from_backend(location['url'], context={})
        image.delete()
        self.assertEqual('deleted', image.status)
        self.assertRaises(glance_store.NotFound, self.store_api.get_from_backend, location['url'], {})

    def test_image_get_data(self):
        image = glance.location.ImageProxy(self.image_stub, {}, self.store_api, self.store_utils)
        self.assertEqual('XXX', image.get_data())

    def test_image_get_data_from_second_location(self):

        def fake_get_from_backend(self, location, offset=0, chunk_size=None, context=None):
            if UUID1 in location:
                raise Exception('not allow download from %s' % location)
            else:
                return self.data[location]
        image1 = glance.location.ImageProxy(self.image_stub, {}, self.store_api, self.store_utils)
        self.assertEqual('XXX', image1.get_data())
        context = glance.context.RequestContext(user=USER1)
        image2, image_stub2 = self._add_image(context, UUID2, 'ZZZ', 3)
        location_data = image2.locations[0]
        with mock.patch('glance.location.store') as mock_store:
            mock_store.get_size_from_uri_and_backend.return_value = 3
            image1.locations.append(location_data)
        self.assertEqual(2, len(image1.locations))
        self.assertEqual(UUID2, location_data['url'])
        self.mock_object(unit_test_utils.FakeStoreAPI, 'get_from_backend', fake_get_from_backend)
        self.assertEqual('ZZZ', image1.get_data().data.fd._source)
        image1.locations.pop(0)
        self.assertEqual(1, len(image1.locations))
        image2.delete()

    def test_image_set_data(self):
        context = glance.context.RequestContext(user=USER1)
        image_stub = ImageStub(UUID2, status='queued', locations=[])
        store_api = unit_test_utils.FakeStoreAPIReader()
        image = glance.location.ImageProxy(image_stub, context, store_api, self.store_utils)
        image.set_data(iter(['YYYY']), 4)
        self.assertEqual(4, image.size)
        self.assertEqual(UUID2, image.locations[0]['url'])
        self.assertEqual('Z', image.checksum)
        self.assertEqual('active', image.status)
        self.assertEqual(4, image.virtual_size)

    def test_image_set_data_inspector_no_match(self):
        context = glance.context.RequestContext(user=USER1)
        image_stub = ImageStub(UUID2, status='queued', locations=[])
        image_stub.disk_format = 'qcow2'
        store_api = unit_test_utils.FakeStoreAPIReader()
        image = glance.location.ImageProxy(image_stub, context, store_api, self.store_utils)
        image.set_data(iter(['YYYY']), 4)
        self.assertEqual(4, image.size)
        self.assertEqual(UUID2, image.locations[0]['url'])
        self.assertEqual('Z', image.checksum)
        self.assertEqual('active', image.status)
        self.assertEqual(0, image.virtual_size)

    @mock.patch('glance.common.format_inspector.QcowInspector.virtual_size', new_callable=mock.PropertyMock)
    @mock.patch('glance.common.format_inspector.QcowInspector.format_match', new_callable=mock.PropertyMock)
    def test_image_set_data_inspector_virtual_size_failure(self, mock_fm, mock_vs):
        mock_fm.return_value = True
        mock_vs.side_effect = ValueError('some error')
        context = glance.context.RequestContext(user=USER1)
        image_stub = ImageStub(UUID2, status='queued', locations=[])
        image_stub.disk_format = 'qcow2'
        store_api = unit_test_utils.FakeStoreAPIReader()
        image = glance.location.ImageProxy(image_stub, context, store_api, self.store_utils)
        image.set_data(iter(['YYYY']), 4)
        self.assertEqual(4, image.size)
        self.assertEqual(UUID2, image.locations[0]['url'])
        self.assertEqual('Z', image.checksum)
        self.assertEqual('active', image.status)
        self.assertEqual(0, image.virtual_size)

    @mock.patch('glance.common.format_inspector.get_inspector')
    def test_image_set_data_inspector_not_needed(self, mock_gi):
        context = glance.context.RequestContext(user=USER1)
        image_stub = ImageStub(UUID2, status='queued', locations=[])
        image_stub.virtual_size = 123
        image_stub.disk_format = 'qcow2'
        store_api = unit_test_utils.FakeStoreAPIReader()
        image = glance.location.ImageProxy(image_stub, context, store_api, self.store_utils)
        image.set_data(iter(['YYYY']), 4)
        self.assertEqual(4, image.size)
        self.assertEqual(UUID2, image.locations[0]['url'])
        self.assertEqual('Z', image.checksum)
        self.assertEqual('active', image.status)
        self.assertEqual(123, image.virtual_size)
        mock_gi.assert_not_called()

    def test_image_set_data_location_metadata(self):
        context = glance.context.RequestContext(user=USER1)
        image_stub = ImageStub(UUID2, status='queued', locations=[])
        loc_meta = {'key': 'value5032'}
        store_api = unit_test_utils.FakeStoreAPI(store_metadata=loc_meta)
        store_utils = unit_test_utils.FakeStoreUtils(store_api)
        image = glance.location.ImageProxy(image_stub, context, store_api, store_utils)
        image.set_data('YYYY', 4)
        self.assertEqual(4, image.size)
        location_data = image.locations[0]
        self.assertEqual(UUID2, location_data['url'])
        self.assertEqual(loc_meta, location_data['metadata'])
        self.assertEqual('Z', image.checksum)
        self.assertEqual('active', image.status)
        image.delete()
        self.assertEqual(image.status, 'deleted')
        self.assertRaises(glance_store.NotFound, self.store_api.get_from_backend, image.locations[0]['url'], {})

    def test_image_set_data_unknown_size(self):
        context = glance.context.RequestContext(user=USER1)
        image_stub = ImageStub(UUID2, status='queued', locations=[])
        image_stub.disk_format = 'iso'
        image = glance.location.ImageProxy(image_stub, context, self.store_api, self.store_utils)
        image.set_data('YYYY', None)
        self.assertEqual(4, image.size)
        self.assertEqual(UUID2, image.locations[0]['url'])
        self.assertEqual('Z', image.checksum)
        self.assertEqual('active', image.status)
        image.delete()
        self.assertEqual(image.status, 'deleted')
        self.assertRaises(glance_store.NotFound, self.store_api.get_from_backend, image.locations[0]['url'], context={})

    @mock.patch('glance.location.LOG')
    def test_image_set_data_valid_signature(self, mock_log):
        context = glance.context.RequestContext(user=USER1)
        extra_properties = {'img_signature_certificate_uuid': 'UUID', 'img_signature_hash_method': 'METHOD', 'img_signature_key_type': 'TYPE', 'img_signature': 'VALID'}
        image_stub = ImageStub(UUID2, status='queued', extra_properties=extra_properties)
        self.mock_object(signature_utils, 'get_verifier', unit_test_utils.fake_get_verifier)
        image = glance.location.ImageProxy(image_stub, context, self.store_api, self.store_utils)
        image.set_data('YYYY', 4)
        self.assertEqual('active', image.status)
        mock_log.info.assert_any_call('Successfully verified signature for image %s', UUID2)

    def test_image_set_data_invalid_signature(self):
        context = glance.context.RequestContext(user=USER1)
        extra_properties = {'img_signature_certificate_uuid': 'UUID', 'img_signature_hash_method': 'METHOD', 'img_signature_key_type': 'TYPE', 'img_signature': 'INVALID'}
        image_stub = ImageStub(UUID2, status='queued', extra_properties=extra_properties)
        self.mock_object(signature_utils, 'get_verifier', unit_test_utils.fake_get_verifier)
        image = glance.location.ImageProxy(image_stub, context, self.store_api, self.store_utils)
        with mock.patch.object(self.store_api, 'delete_from_backend') as mock_delete:
            self.assertRaises(cursive_exception.SignatureVerificationError, image.set_data, 'YYYY', 4)
            mock_delete.assert_called()

    def test_image_set_data_invalid_signature_missing_metadata(self):
        context = glance.context.RequestContext(user=USER1)
        extra_properties = {'img_signature_hash_method': 'METHOD', 'img_signature_key_type': 'TYPE', 'img_signature': 'INVALID'}
        image_stub = ImageStub(UUID2, status='queued', extra_properties=extra_properties)
        self.mock_object(signature_utils, 'get_verifier', unit_test_utils.fake_get_verifier)
        image = glance.location.ImageProxy(image_stub, context, self.store_api, self.store_utils)
        image.set_data('YYYY', 4)
        self.assertEqual(UUID2, image.locations[0]['url'])
        self.assertEqual('Z', image.checksum)
        self.assertEqual('active', image.status)

    def _add_image(self, context, image_id, data, len):
        image_stub = ImageStub(image_id, status='queued', locations=[])
        image = glance.location.ImageProxy(image_stub, context, self.store_api, self.store_utils)
        image.set_data(data, len)
        self.assertEqual(len, image.size)
        location = {'url': image_id, 'metadata': {}, 'status': 'active'}
        self.assertEqual([location], image.locations)
        self.assertEqual([location], image_stub.locations)
        self.assertEqual('active', image.status)
        return (image, image_stub)

    def test_image_change_append_invalid_location_uri(self):
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        location_bad = {'url': 'unknown://location', 'metadata': {}}
        self.assertRaises(exception.BadStoreUri, image1.locations.append, location_bad)
        image1.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())

    def test_image_change_append_invalid_location_metatdata(self):
        UUID3 = 'a8a61ec4-d7a3-11e2-8c28-000c29c27581'
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        image2, image_stub2 = self._add_image(context, UUID3, 'YYYY', 4)
        location_bad = {'url': UUID3, 'metadata': b'a invalid metadata'}
        self.assertRaises(glance_store.BackendException, image1.locations.append, location_bad)
        image1.delete()
        image2.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())
        self.assertNotIn(UUID3, self.store_api.data.keys())

    def test_image_change_append_locations(self):
        UUID3 = 'a8a61ec4-d7a3-11e2-8c28-000c29c27581'
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        image2, image_stub2 = self._add_image(context, UUID3, 'YYYY', 4)
        location2 = {'url': UUID2, 'metadata': {}, 'status': 'active'}
        location3 = {'url': UUID3, 'metadata': {}, 'status': 'active'}
        image1.locations.append(location3)
        self.assertEqual([location2, location3], image_stub1.locations)
        self.assertEqual([location2, location3], image1.locations)
        image1.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())
        self.assertNotIn(UUID3, self.store_api.data.keys())
        image2.delete()

    def test_image_change_pop_location(self):
        UUID3 = 'a8a61ec4-d7a3-11e2-8c28-000c29c27581'
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        image2, image_stub2 = self._add_image(context, UUID3, 'YYYY', 4)
        location2 = {'url': UUID2, 'metadata': {}, 'status': 'active'}
        location3 = {'url': UUID3, 'metadata': {}, 'status': 'active'}
        image1.locations.append(location3)
        self.assertEqual([location2, location3], image_stub1.locations)
        self.assertEqual([location2, location3], image1.locations)
        image1.locations.pop()
        self.assertEqual([location2], image_stub1.locations)
        self.assertEqual([location2], image1.locations)
        image1.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())
        self.assertNotIn(UUID3, self.store_api.data.keys())
        image2.delete()

    def test_image_change_extend_invalid_locations_uri(self):
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        location_bad = {'url': 'unknown://location', 'metadata': {}}
        self.assertRaises(exception.BadStoreUri, image1.locations.extend, [location_bad])
        image1.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())

    def test_image_change_extend_invalid_locations_metadata(self):
        UUID3 = 'a8a61ec4-d7a3-11e2-8c28-000c29c27581'
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        image2, image_stub2 = self._add_image(context, UUID3, 'YYYY', 4)
        location_bad = {'url': UUID3, 'metadata': b'a invalid metadata'}
        self.assertRaises(glance_store.BackendException, image1.locations.extend, [location_bad])
        image1.delete()
        image2.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())
        self.assertNotIn(UUID3, self.store_api.data.keys())

    def test_image_change_extend_locations(self):
        UUID3 = 'a8a61ec4-d7a3-11e2-8c28-000c29c27581'
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        image2, image_stub2 = self._add_image(context, UUID3, 'YYYY', 4)
        location2 = {'url': UUID2, 'metadata': {}, 'status': 'active'}
        location3 = {'url': UUID3, 'metadata': {}, 'status': 'active'}
        image1.locations.extend([location3])
        self.assertEqual([location2, location3], image_stub1.locations)
        self.assertEqual([location2, location3], image1.locations)
        image1.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())
        self.assertNotIn(UUID3, self.store_api.data.keys())
        image2.delete()

    def test_image_change_remove_location(self):
        UUID3 = 'a8a61ec4-d7a3-11e2-8c28-000c29c27581'
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        image2, image_stub2 = self._add_image(context, UUID3, 'YYYY', 4)
        location2 = {'url': UUID2, 'metadata': {}, 'status': 'active'}
        location3 = {'url': UUID3, 'metadata': {}, 'status': 'active'}
        location_bad = {'url': 'unknown://location', 'metadata': {}}
        image1.locations.extend([location3])
        image1.locations.remove(location2)
        self.assertEqual([location3], image_stub1.locations)
        self.assertEqual([location3], image1.locations)
        self.assertRaises(ValueError, image1.locations.remove, location_bad)
        image1.delete()
        image2.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())
        self.assertNotIn(UUID3, self.store_api.data.keys())

    def test_image_change_delete_location(self):
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        del image1.locations[0]
        self.assertEqual([], image_stub1.locations)
        self.assertEqual(0, len(image1.locations))
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())
        image1.delete()

    def test_image_change_insert_invalid_location_uri(self):
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        location_bad = {'url': 'unknown://location', 'metadata': {}}
        self.assertRaises(exception.BadStoreUri, image1.locations.insert, 0, location_bad)
        image1.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())

    def test_image_change_insert_invalid_location_metadata(self):
        UUID3 = 'a8a61ec4-d7a3-11e2-8c28-000c29c27581'
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        image2, image_stub2 = self._add_image(context, UUID3, 'YYYY', 4)
        location_bad = {'url': UUID3, 'metadata': b'a invalid metadata'}
        self.assertRaises(glance_store.BackendException, image1.locations.insert, 0, location_bad)
        image1.delete()
        image2.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())
        self.assertNotIn(UUID3, self.store_api.data.keys())

    def test_image_change_insert_location(self):
        UUID3 = 'a8a61ec4-d7a3-11e2-8c28-000c29c27581'
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        image2, image_stub2 = self._add_image(context, UUID3, 'YYYY', 4)
        location2 = {'url': UUID2, 'metadata': {}, 'status': 'active'}
        location3 = {'url': UUID3, 'metadata': {}, 'status': 'active'}
        image1.locations.insert(0, location3)
        self.assertEqual([location3, location2], image_stub1.locations)
        self.assertEqual([location3, location2], image1.locations)
        image1.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())
        self.assertNotIn(UUID3, self.store_api.data.keys())
        image2.delete()

    def test_image_change_delete_locations(self):
        UUID3 = 'a8a61ec4-d7a3-11e2-8c28-000c29c27581'
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        image2, image_stub2 = self._add_image(context, UUID3, 'YYYY', 4)
        location2 = {'url': UUID2, 'metadata': {}}
        location3 = {'url': UUID3, 'metadata': {}}
        image1.locations.insert(0, location3)
        del image1.locations[0:100]
        self.assertEqual([], image_stub1.locations)
        self.assertEqual(0, len(image1.locations))
        self.assertRaises(exception.BadStoreUri, image1.locations.insert, 0, location2)
        self.assertRaises(exception.BadStoreUri, image2.locations.insert, 0, location3)
        image1.delete()
        image2.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())
        self.assertNotIn(UUID3, self.store_api.data.keys())

    def test_image_change_adding_invalid_location_uri(self):
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image_stub1 = ImageStub('fake_image_id', status='queued', locations=[])
        image1 = glance.location.ImageProxy(image_stub1, context, self.store_api, self.store_utils)
        location_bad = {'url': 'unknown://location', 'metadata': {}}
        self.assertRaises(exception.BadStoreUri, image1.locations.__iadd__, [location_bad])
        self.assertEqual([], image_stub1.locations)
        self.assertEqual([], image1.locations)
        image1.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())

    def test_image_change_adding_invalid_location_metadata(self):
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        image_stub2 = ImageStub('fake_image_id', status='queued', locations=[])
        image2 = glance.location.ImageProxy(image_stub2, context, self.store_api, self.store_utils)
        location_bad = {'url': UUID2, 'metadata': b'a invalid metadata'}
        self.assertRaises(glance_store.BackendException, image2.locations.__iadd__, [location_bad])
        self.assertEqual([], image_stub2.locations)
        self.assertEqual([], image2.locations)
        image1.delete()
        image2.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())

    def test_image_change_adding_locations(self):
        UUID3 = 'a8a61ec4-d7a3-11e2-8c28-000c29c27581'
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        image2, image_stub2 = self._add_image(context, UUID3, 'YYYY', 4)
        image_stub3 = ImageStub('fake_image_id', status='queued', locations=[])
        image3 = glance.location.ImageProxy(image_stub3, context, self.store_api, self.store_utils)
        location2 = {'url': UUID2, 'metadata': {}}
        location3 = {'url': UUID3, 'metadata': {}}
        with mock.patch('glance.location.store') as mock_store:
            mock_store.get_size_from_uri_and_backend.return_value = 4
            image3.locations += [location2, location3]
        self.assertEqual([location2, location3], image_stub3.locations)
        self.assertEqual([location2, location3], image3.locations)
        image3.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())
        self.assertNotIn(UUID3, self.store_api.data.keys())
        image1.delete()
        image2.delete()

    def test_image_get_location_index(self):
        UUID3 = 'a8a61ec4-d7a3-11e2-8c28-000c29c27581'
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        image2, image_stub2 = self._add_image(context, UUID3, 'YYYY', 4)
        image_stub3 = ImageStub('fake_image_id', status='queued', locations=[])
        image3 = glance.location.ImageProxy(image_stub3, context, self.store_api, self.store_utils)
        location2 = {'url': UUID2, 'metadata': {}}
        location3 = {'url': UUID3, 'metadata': {}}
        with mock.patch('glance.location.store') as mock_store:
            mock_store.get_size_from_uri_and_backend.return_value = 4
            image3.locations += [location2, location3]
        self.assertEqual(1, image_stub3.locations.index(location3))
        image3.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())
        self.assertNotIn(UUID3, self.store_api.data.keys())
        image1.delete()
        image2.delete()

    def test_image_get_location_by_index(self):
        UUID3 = 'a8a61ec4-d7a3-11e2-8c28-000c29c27581'
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        image2, image_stub2 = self._add_image(context, UUID3, 'YYYY', 4)
        image_stub3 = ImageStub('fake_image_id', status='queued', locations=[])
        image3 = glance.location.ImageProxy(image_stub3, context, self.store_api, self.store_utils)
        location2 = {'url': UUID2, 'metadata': {}}
        location3 = {'url': UUID3, 'metadata': {}}
        with mock.patch('glance.location.store') as mock_store:
            mock_store.get_size_from_uri_and_backend.return_value = 4
            image3.locations += [location2, location3]
        self.assertEqual(1, image_stub3.locations.index(location3))
        self.assertEqual(location2, image_stub3.locations[0])
        image3.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())
        self.assertNotIn(UUID3, self.store_api.data.keys())
        image1.delete()
        image2.delete()

    def test_image_checking_location_exists(self):
        UUID3 = 'a8a61ec4-d7a3-11e2-8c28-000c29c27581'
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        image2, image_stub2 = self._add_image(context, UUID3, 'YYYY', 4)
        image_stub3 = ImageStub('fake_image_id', status='queued', locations=[])
        image3 = glance.location.ImageProxy(image_stub3, context, self.store_api, self.store_utils)
        location2 = {'url': UUID2, 'metadata': {}}
        location3 = {'url': UUID3, 'metadata': {}}
        location_bad = {'url': 'unknown://location', 'metadata': {}}
        with mock.patch('glance.location.store') as mock_store:
            mock_store.get_size_from_uri_and_backend.return_value = 4
            image3.locations += [location2, location3]
        self.assertIn(location3, image_stub3.locations)
        self.assertNotIn(location_bad, image_stub3.locations)
        image3.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())
        self.assertNotIn(UUID3, self.store_api.data.keys())
        image1.delete()
        image2.delete()

    def test_image_reverse_locations_order(self):
        UUID3 = 'a8a61ec4-d7a3-11e2-8c28-000c29c27581'
        self.assertEqual(2, len(self.store_api.data.keys()))
        context = glance.context.RequestContext(user=USER1)
        image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
        image2, image_stub2 = self._add_image(context, UUID3, 'YYYY', 4)
        location2 = {'url': UUID2, 'metadata': {}}
        location3 = {'url': UUID3, 'metadata': {}}
        image_stub3 = ImageStub('fake_image_id', status='queued', locations=[])
        image3 = glance.location.ImageProxy(image_stub3, context, self.store_api, self.store_utils)
        with mock.patch('glance.location.store') as mock_store:
            mock_store.get_size_from_uri_and_backend.return_value = 4
            image3.locations += [location2, location3]
        image_stub3.locations.reverse()
        self.assertEqual([location3, location2], image_stub3.locations)
        self.assertEqual([location3, location2], image3.locations)
        image3.delete()
        self.assertEqual(2, len(self.store_api.data.keys()))
        self.assertNotIn(UUID2, self.store_api.data.keys())
        self.assertNotIn(UUID3, self.store_api.data.keys())
        image1.delete()
        image2.delete()
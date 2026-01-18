import hashlib
import io
from unittest import mock
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import rbd as rbd_store
from glance_store import exceptions
from glance_store import location as g_location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils as test_utils
class TestReSize(base.StoreBaseTest, test_store_capabilities.TestStoreCapabilitiesChecking):

    def setUp(self):
        """Establish a clean test environment."""
        super(TestReSize, self).setUp()
        rbd_store.rados = MockRados
        rbd_store.rbd = MockRBD
        self.store = rbd_store.Store(self.conf)
        self.store.configure()
        self.store_specs = {'pool': 'fake_pool', 'image': 'fake_image', 'snapshot': 'fake_snapshot'}
        self.location = rbd_store.StoreLocation(self.store_specs, self.conf)
        self.hash_algo = 'sha256'

    def test_add_w_image_size_zero_less_resizes(self):
        """Assert that correct size is returned even though 0 was provided."""
        data_len = 57 * units.Mi
        data_iter = test_utils.FakeData(data_len)
        with mock.patch.object(rbd_store.rbd.Image, 'resize') as resize:
            with mock.patch.object(rbd_store.rbd.Image, 'write') as write:
                ret = self.store.add('fake_image_id', data_iter, 0, self.hash_algo)
                expected = 1
                expected_calls = []
                data_len_temp = data_len
                resize_amount = self.store.WRITE_CHUNKSIZE
                while data_len_temp > 0:
                    resize_amount *= 2
                    expected_calls.append(resize_amount + (data_len - data_len_temp))
                    data_len_temp -= resize_amount
                    expected += 1
                self.assertEqual(expected, resize.call_count)
                resize.assert_has_calls([mock.call(call) for call in expected_calls])
                expected = [self.store.WRITE_CHUNKSIZE for i in range(int(data_len / self.store.WRITE_CHUNKSIZE))] + [data_len % self.store.WRITE_CHUNKSIZE]
                actual = [len(args[0]) for args, kwargs in write.call_args_list]
                self.assertEqual(expected, actual)
                self.assertEqual(data_len, resize.call_args_list[-1][0][0])
                self.assertEqual(data_len, ret[1])

    def test_resize_on_write_ceiling(self):
        image = mock.MagicMock()
        ret = self.store._resize_on_write(image, 32, 16, 16)
        self.assertEqual(0, ret)
        image.resize.assert_not_called()
        self.store.size = 8
        ret = self.store._resize_on_write(image, 0, 16, 16)
        self.assertEqual(8 + self.store.WRITE_CHUNKSIZE * 2, ret)
        self.assertEqual(self.store.WRITE_CHUNKSIZE * 2, self.store.resize_amount)
        image.resize.assert_called_once_with(ret)
        image.resize.reset_mock()
        self.store.size = ret
        ret = self.store._resize_on_write(image, 0, 64, 16)
        self.assertEqual(8 + self.store.WRITE_CHUNKSIZE * 2, ret)
        image.resize.assert_not_called()
        ret = self.store._resize_on_write(image, 0, ret + 1, 16)
        self.assertEqual(8 + self.store.WRITE_CHUNKSIZE * 6, ret)
        image.resize.assert_called_once_with(ret)
        self.assertEqual(self.store.WRITE_CHUNKSIZE * 4, self.store.resize_amount)
        image.resize.reset_mock()
        self.store.resize_amount = 2 * units.Gi
        self.store.size = 1 * units.Gi
        ret = self.store._resize_on_write(image, 0, 4097 * units.Mi, 16)
        self.assertEqual(4 * units.Gi, self.store.resize_amount)
        self.assertEqual((1 + 4) * units.Gi, ret)
        self.store.size = ret
        ret = self.store._resize_on_write(image, 0, 6144 * units.Mi, 16)
        self.assertEqual(8 * units.Gi, self.store.resize_amount)
        self.assertEqual((1 + 4 + 8) * units.Gi, ret)
        self.store.size = ret
        ret = self.store._resize_on_write(image, 0, 14336 * units.Mi, 16)
        self.assertEqual(8 * units.Gi, self.store.resize_amount)
        self.assertEqual((1 + 4 + 8 + 8) * units.Gi, ret)
        self.store.size = ret
        ret = self.store._resize_on_write(image, 0, 22528 * units.Mi, 16)
        self.assertEqual(8 * units.Gi, self.store.resize_amount)
        self.assertEqual((1 + 4 + 8 + 8 + 8) * units.Gi, ret)
        image.resize.assert_has_calls([mock.call(5 * units.Gi), mock.call(13 * units.Gi), mock.call(21 * units.Gi), mock.call(29 * units.Gi)])
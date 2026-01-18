from unittest import mock
from glance_store import backend
from glance_store import exceptions
from glance_store.tests import base
def _good_metadata(self, in_metadata):
    mstore = mock.Mock()
    mstore.add.return_value = (self.location, self.size, self.checksum, in_metadata)
    location, size, checksum, metadata = backend.store_add_to_backend(self.image_id, self.data, self.size, mstore)
    mstore.add.assert_called_once_with(self.image_id, mock.ANY, self.size, context=None, verifier=None)
    self.assertEqual(self.location, location)
    self.assertEqual(self.size, size)
    self.assertEqual(self.checksum, checksum)
    self.assertEqual(in_metadata, metadata)
    newstore = mock.Mock()
    newstore.add.return_value = (self.location, self.size, self.checksum, self.multihash, in_metadata)
    location, size, checksum, multihash, metadata = backend.store_add_to_backend_with_multihash(self.image_id, self.data, self.size, self.hash_algo, newstore)
    newstore.add.assert_called_once_with(self.image_id, mock.ANY, self.size, self.hash_algo, context=None, verifier=None)
    self.assertEqual(self.location, location)
    self.assertEqual(self.size, size)
    self.assertEqual(self.checksum, checksum)
    self.assertEqual(self.multihash, multihash)
    self.assertEqual(in_metadata, metadata)
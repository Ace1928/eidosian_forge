import io
from os import environ
import glance_store
import os_client_config
from oslo_config import cfg
import testtools
class BaseFunctionalTests(Base):

    def test_add(self):
        image_file = io.BytesIO(IMAGE_BITS)
        loc, written, _, _ = self.store.add(UUID1, image_file, len(IMAGE_BITS))
        self.assertEqual(len(IMAGE_BITS), written)

    def test_delete(self):
        image_file = io.BytesIO(IMAGE_BITS)
        loc, written, _, _ = self.store.add(UUID2, image_file, len(IMAGE_BITS))
        location = glance_store.location.get_location_from_uri(loc)
        self.store.delete(location)

    def test_get_size(self):
        image_file = io.BytesIO(IMAGE_BITS)
        loc, written, _, _ = self.store.add(UUID3, image_file, len(IMAGE_BITS))
        location = glance_store.location.get_location_from_uri(loc)
        size = self.store.get_size(location)
        self.assertEqual(len(IMAGE_BITS), size)

    def test_get(self):
        image_file = io.BytesIO(IMAGE_BITS)
        loc, written, _, _ = self.store.add(UUID3, image_file, len(IMAGE_BITS))
        location = glance_store.location.get_location_from_uri(loc)
        image, size = self.store.get(location)
        self.assertEqual(len(IMAGE_BITS), size)
        data = b''
        for chunk in image:
            data += chunk
        self.assertEqual(IMAGE_BITS, data)
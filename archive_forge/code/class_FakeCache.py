from unittest import mock
import webob
from glance.api.v2 import cached_images
import glance.gateway
from glance import image_cache
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class FakeCache(image_cache.ImageCache):

    def __init__(self):
        self.init_driver()
        self.deleted_images = []

    def init_driver(self):
        pass

    def get_cached_images(self):
        return [{'image_id': 'test'}]

    def delete_cached_image(self, image_id):
        self.deleted_images.append(image_id)

    def delete_all_cached_images(self):
        self.delete_cached_image(self.get_cached_images()[0].get('image_id'))
        return 1

    def get_queued_images(self):
        return {'test': 'passed'}

    def queue_image(self, image_id):
        return 'pass'

    def delete_queued_image(self, image_id):
        self.deleted_images.append(image_id)

    def delete_all_queued_images(self):
        self.delete_queued_image('deleted_img')
        return 1
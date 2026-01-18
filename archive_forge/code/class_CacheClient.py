import os
from oslo_serialization import jsonutils as json
from glance.common import client as base_client
from glance.common import exception
from glance.i18n import _
class CacheClient(base_client.BaseClient):
    DEFAULT_PORT = 9292
    DEFAULT_DOC_ROOT = '/v2'

    def delete_cached_image(self, image_id):
        """
        Delete a specified image from the cache
        """
        self.do_request('DELETE', '/cached_images/%s' % image_id)
        return True

    def get_cached_images(self, **kwargs):
        """
        Returns a list of images stored in the image cache.
        """
        res = self.do_request('GET', '/cached_images')
        data = json.loads(res.read())['cached_images']
        return data

    def get_queued_images(self, **kwargs):
        """
        Returns a list of images queued for caching
        """
        res = self.do_request('GET', '/queued_images')
        data = json.loads(res.read())['queued_images']
        return data

    def delete_all_cached_images(self):
        """
        Delete all cached images
        """
        res = self.do_request('DELETE', '/cached_images')
        data = json.loads(res.read())
        num_deleted = data['num_deleted']
        return num_deleted

    def queue_image_for_caching(self, image_id):
        """
        Queue an image for prefetching into cache
        """
        self.do_request('PUT', '/queued_images/%s' % image_id)
        return True

    def delete_queued_image(self, image_id):
        """
        Delete a specified image from the cache queue
        """
        self.do_request('DELETE', '/queued_images/%s' % image_id)
        return True

    def delete_all_queued_images(self):
        """
        Delete all queued images
        """
        res = self.do_request('DELETE', '/queued_images')
        data = json.loads(res.read())
        num_deleted = data['num_deleted']
        return num_deleted
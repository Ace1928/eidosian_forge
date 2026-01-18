import calendar
import time
import eventlet
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_log import versionutils
from oslo_utils import encodeutils
from glance.common import crypt
from glance.common import exception
from glance.common import timeutils
from glance import context
import glance.db as db_api
from glance.i18n import _, _LC, _LE, _LI, _LW
class ScrubDBQueue(object):
    """Database-based image scrub queue class."""

    def __init__(self):
        self.scrub_time = CONF.scrub_time
        self.metadata_encryption_key = CONF.metadata_encryption_key
        self.admin_context = context.get_admin_context(show_deleted=True)

    def add_location(self, image_id, location):
        """Adding image location to scrub queue.

        :param image_id: The opaque image identifier
        :param location: The opaque image location

        :returns: A boolean value to indicate success or not
        """
        loc_id = location.get('id')
        if loc_id:
            db_api.get_api().image_location_delete(self.admin_context, image_id, loc_id, 'pending_delete')
            return True
        else:
            return False

    def _get_images_page(self, marker):
        filters = {'deleted': True, 'status': 'pending_delete'}
        return db_api.get_api().image_get_all(self.admin_context, filters=filters, marker=marker, limit=REASONABLE_DB_PAGE_SIZE)

    def _get_all_images(self):
        """Generator to fetch all appropriate images, paging as needed."""
        marker = None
        while True:
            images = self._get_images_page(marker)
            if len(images) == 0:
                break
            marker = images[-1]['id']
            for image in images:
                yield image

    def get_all_locations(self):
        """Returns a list of image id and location tuple from scrub queue.

        :returns: a list of image id, location id and uri tuple from
            scrub queue

        """
        ret = []
        for image in self._get_all_images():
            deleted_at = image.get('deleted_at')
            if not deleted_at:
                continue
            deleted_at = timeutils.isotime(deleted_at)
            date_str = deleted_at.rsplit('.', 1)[0].rsplit(',', 1)[0]
            delete_time = calendar.timegm(time.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ'))
            if delete_time + self.scrub_time > time.time():
                continue
            for loc in image['locations']:
                if loc['status'] != 'pending_delete':
                    continue
                if self.metadata_encryption_key:
                    uri = crypt.urlsafe_decrypt(self.metadata_encryption_key, loc['url'])
                else:
                    uri = loc['url']
                backend = loc['metadata'].get('store')
                if CONF.enabled_backends:
                    ret.append((image['id'], loc['id'], uri, backend))
                else:
                    ret.append((image['id'], loc['id'], uri))
        return ret

    def has_image(self, image_id):
        """Returns whether the queue contains an image or not.

        :param image_id: The opaque image identifier

        :returns: a boolean value to inform including or not
        """
        try:
            image = db_api.get_api().image_get(self.admin_context, image_id)
            return image['status'] == 'pending_delete'
        except exception.NotFound:
            return False
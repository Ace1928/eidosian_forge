import copy
import functools
import json
import os
import urllib.request
import glance_store as store_api
from glance_store import backend
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import timeutils
from oslo_utils import units
import taskflow
from taskflow.patterns import linear_flow as lf
from taskflow import retry
from taskflow import task
from glance.api import common as api_common
import glance.async_.flows._internal_plugins as internal_plugins
import glance.async_.flows.plugins as import_plugins
from glance.async_ import utils
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance.common.scripts import utils as script_utils
from glance.common import store_utils
from glance.i18n import _, _LE, _LI
from glance.quota import keystone as ks_quota
class _ImportActions(object):
    """Actions available for being performed on an image during import.

    This defines the available actions that can be performed on an image
    during import, which may be done with an image owned by another user.

    Do not instantiate this object directly, get it from ImportActionWrapper.
    """
    IMPORTING_STORES_KEY = 'os_glance_importing_to_stores'
    IMPORT_FAILED_KEY = 'os_glance_failed_import'

    def __init__(self, image):
        self._image = image

    @property
    def image_id(self):
        return self._image.image_id

    @property
    def image_size(self):
        return self._image.size

    @property
    def image_locations(self):
        return copy.deepcopy(self._image.locations)

    @property
    def image_disk_format(self):
        return self._image.disk_format

    @property
    def image_container_format(self):
        return self._image.container_format

    @property
    def image_extra_properties(self):
        return dict(self._image.extra_properties)

    @property
    def image_status(self):
        return self._image.status

    def merge_store_list(self, list_key, stores, subtract=False):
        stores = set([store for store in stores if store])
        existing = set(self._image.extra_properties.get(list_key, '').split(','))
        if subtract:
            if stores - existing:
                LOG.debug('Stores %(stores)s not in %(key)s for image %(image_id)s', {'stores': ','.join(sorted(stores - existing)), 'key': list_key, 'image_id': self.image_id})
            merged_stores = existing - stores
        else:
            merged_stores = existing | stores
        stores_list = ','.join(sorted((store for store in merged_stores if store)))
        self._image.extra_properties[list_key] = stores_list
        LOG.debug('Image %(image_id)s %(key)s=%(stores)s', {'image_id': self.image_id, 'key': list_key, 'stores': stores_list})

    def add_importing_stores(self, stores):
        """Add a list of stores to the importing list.

        Add stores to os_glance_importing_to_stores

        :param stores: A list of store names
        """
        self.merge_store_list(self.IMPORTING_STORES_KEY, stores)

    def remove_importing_stores(self, stores):
        """Remove a list of stores from the importing list.

        Remove stores from os_glance_importing_to_stores

        :param stores: A list of store names
        """
        self.merge_store_list(self.IMPORTING_STORES_KEY, stores, subtract=True)

    def add_failed_stores(self, stores):
        """Add a list of stores to the failed list.

        Add stores to os_glance_failed_import

        :param stores: A list of store names
        """
        self.merge_store_list(self.IMPORT_FAILED_KEY, stores)

    def remove_failed_stores(self, stores):
        """Remove a list of stores from the failed list.

        Remove stores from os_glance_failed_import

        :param stores: A list of store names
        """
        self.merge_store_list(self.IMPORT_FAILED_KEY, stores, subtract=True)

    def set_image_data(self, uri, task_id, backend, set_active, callback=None):
        """Populate image with data on a specific backend.

        This is used during an image import operation to populate the data
        in a given store for the image. If this object wraps an admin-capable
        image_repo, then this will be done with admin credentials on behalf
        of a user already determined to be able to perform this operation
        (such as a copy-image import of an existing image owned by another
        user).

        :param uri: Source URL for image data
        :param task_id: The task responsible for this operation
        :param backend: The backend store to target the data
        :param set_active: Whether or not to set the image to 'active'
                           state after the operation completes
        :param callback: A callback function with signature:
                         fn(action, chunk_bytes, total_bytes)
                         which should be called while processing the image
                         approximately every minute.
        """
        if callback:
            callback = functools.partial(callback, self)
        return image_import.set_image_data(self._image, uri, task_id, backend=backend, set_active=set_active, callback=callback)

    def set_image_attribute(self, **attrs):
        """Set an image attribute.

        This allows setting various image attributes which will be saved
        upon exiting the ImportActionWrapper context.

        :param attrs: kwarg list of attributes to set on the image
        :raises: AttributeError if an attribute outside the set of allowed
                 ones is present in attrs.
        """
        allowed = ['status', 'disk_format', 'container_format', 'virtual_size', 'size']
        for attr, value in attrs.items():
            if attr not in allowed:
                raise AttributeError('Setting %s is not allowed' % attr)
            setattr(self._image, attr, value)

    def set_image_extra_properties(self, properties):
        """Merge values into image extra_properties.

        This allows a plugin to set additional properties on the image,
        as long as those are outside the reserved namespace. Any keys
        in the internal namespace will be dropped (and logged).

        :param properties: A dict of properties to be merged in
        """
        for key, value in properties.items():
            if key.startswith(api_common.GLANCE_RESERVED_NS):
                LOG.warning('Dropping %(key)s=%(val)s during metadata injection for %(image)s', {'key': key, 'val': value, 'image': self.image_id})
            else:
                self._image.extra_properties[key] = value

    def remove_location_for_store(self, backend):
        """Remove a location from an image given a backend store.

        Given a backend store, remove the corresponding location from the
        image's set of locations. If the last location is removed, remove
        the image checksum, hash information, and size.

        :param backend: The backend store to remove from the image
        """
        for i, location in enumerate(self._image.locations):
            if location.get('metadata', {}).get('store') == backend:
                try:
                    self._image.locations.pop(i)
                except (store_exceptions.NotFound, store_exceptions.Forbidden):
                    msg = _('Error deleting from store %(store)s when reverting.') % {'store': backend}
                    LOG.warning(msg)
                except Exception:
                    msg = _('Unexpected exception when deleting from store %(store)s.') % {'store': backend}
                    LOG.warning(msg)
                else:
                    if len(self._image.locations) == 0:
                        self._image.checksum = None
                        self._image.os_hash_algo = None
                        self._image.os_hash_value = None
                        self._image.size = None
                break

    def pop_extra_property(self, name):
        """Delete the named extra_properties value, if present.

        If the image.extra_properties dict contains the named key,
        delete it.
        :param name: The key to delete.
        """
        self._image.extra_properties.pop(name, None)
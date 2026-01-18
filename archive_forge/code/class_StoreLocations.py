from collections import abc
import copy
import functools
from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store as store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from glance.common import exception
from glance.common import format_inspector
from glance.common import store_utils
from glance.common import utils
import glance.domain.proxy
from glance.i18n import _, _LE, _LI, _LW
@functools.total_ordering
class StoreLocations(abc.MutableSequence):
    """
    The proxy for store location property. It takes responsibility for::

        1. Location uri correctness checking when adding a new location.
        2. Remove the image data from the store when a location is removed
           from an image.

    """

    def __init__(self, image_proxy, value):
        self.image_proxy = image_proxy
        if isinstance(value, list):
            self.value = value
        else:
            self.value = list(value)

    def append(self, location):
        self.insert(len(self.value), location)

    def extend(self, other):
        if isinstance(other, StoreLocations):
            locations = other.value
        else:
            locations = list(other)
        for location in locations:
            self.append(location)

    def insert(self, i, location):
        _check_image_location(self.image_proxy.context, self.image_proxy.store_api, self.image_proxy.store_utils, location)
        location['status'] = 'active'
        if _count_duplicated_locations(self.value, location) > 0:
            raise exception.DuplicateLocation(location=location['url'])
        self.value.insert(i, location)
        _set_image_size(self.image_proxy.context, self.image_proxy, [location])

    def pop(self, i=-1):
        location = self.value.pop(i)
        try:
            self.image_proxy.store_utils.delete_image_location_from_backend(self.image_proxy.context, self.image_proxy.image.image_id, location)
        except store.exceptions.NotFound:
            with excutils.save_and_reraise_exception():
                pass
        except Exception:
            with excutils.save_and_reraise_exception():
                self.value.insert(i, location)
        return location

    def count(self, location):
        return self.value.count(location)

    def index(self, location, *args):
        return self.value.index(location, *args)

    def remove(self, location):
        if self.count(location):
            self.pop(self.index(location))
        else:
            self.value.remove(location)

    def reverse(self):
        self.value.reverse()
    __hash__ = None

    def __getitem__(self, i):
        return self.value.__getitem__(i)

    def __setitem__(self, i, location):
        _check_image_location(self.image_proxy.context, self.image_proxy.store_api, self.image_proxy.store_utils, location)
        location['status'] = 'active'
        self.value.__setitem__(i, location)
        _set_image_size(self.image_proxy.context, self.image_proxy, [location])

    def __delitem__(self, i):
        if isinstance(i, slice):
            if i.step not in (None, 1):
                raise NotImplementedError('slice with step')
            self.__delslice__(i.start, i.stop)
            return
        location = None
        try:
            location = self.value[i]
        except Exception:
            del self.value[i]
            return
        self.image_proxy.store_utils.delete_image_location_from_backend(self.image_proxy.context, self.image_proxy.image.image_id, location)
        del self.value[i]

    def __delslice__(self, i, j):
        i = 0 if i is None else max(i, 0)
        j = len(self) if j is None else max(j, 0)
        locations = []
        try:
            locations = self.value[i:j]
        except Exception:
            del self.value[i:j]
            return
        for location in locations:
            self.image_proxy.store_utils.delete_image_location_from_backend(self.image_proxy.context, self.image_proxy.image.image_id, location)
            del self.value[i]

    def __iadd__(self, other):
        self.extend(other)
        return self

    def __contains__(self, location):
        return location in self.value

    def __len__(self):
        return len(self.value)

    def __cast(self, other):
        if isinstance(other, StoreLocations):
            return other.value
        else:
            return other

    def __eq__(self, other):
        return self.value == self.__cast(other)

    def __lt__(self, other):
        return self.value < self.__cast(other)

    def __iter__(self):
        return iter(self.value)

    def __copy__(self):
        return type(self)(self.image_proxy, self.value)

    def __deepcopy__(self, memo):
        value = copy.deepcopy(self.value, memo)
        self.image_proxy.image.locations = value
        return type(self)(self.image_proxy, value)
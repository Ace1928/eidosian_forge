import secrets
from dogpile.cache import region
from dogpile.cache import util
from oslo_cache import core as cache
from keystone.common.cache import _context_cache
import keystone.conf
class RegionInvalidationManager(object):
    REGION_KEY_PREFIX = '<<<region>>>:'

    def __init__(self, invalidation_region, region_name):
        self._invalidation_region = invalidation_region
        self._region_key = self.REGION_KEY_PREFIX + region_name

    def _generate_new_id(self):
        return secrets.token_bytes(10)

    @property
    def region_id(self):
        return self._invalidation_region.get_or_create(self._region_key, self._generate_new_id, expiration_time=-1)

    def invalidate_region(self):
        new_region_id = self._generate_new_id()
        self._invalidation_region.set(self._region_key, new_region_id)
        return new_region_id

    def is_region_key(self, key):
        return key == self._region_key
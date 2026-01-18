from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemcacheVersionValueValuesEnum(_messages.Enum):
    """Output only. Major version of memcached server running on this node,
    e.g. MEMCACHE_1_5

    Values:
      MEMCACHE_VERSION_UNSPECIFIED: Memcache version is not specified by
        customer
      MEMCACHE_1_5: Memcached 1.5 version.
      MEMCACHE_1_6_15: Memcached 1.6.15 version.
    """
    MEMCACHE_VERSION_UNSPECIFIED = 0
    MEMCACHE_1_5 = 1
    MEMCACHE_1_6_15 = 2
import logging
from google.appengine.api import memcache
from . import base
from ..discovery_cache import DISCOVERY_DOC_MAX_AGE
Constructor.

        Args:
          max_age: Cache expiration in seconds.
        
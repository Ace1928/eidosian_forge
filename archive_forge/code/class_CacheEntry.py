import base64
import pickle
from datetime import datetime, timezone
from django.conf import settings
from django.core.cache.backends.base import DEFAULT_TIMEOUT, BaseCache
from django.db import DatabaseError, connections, models, router, transaction
from django.utils.timezone import now as tz_now
class CacheEntry:
    _meta = Options(table)
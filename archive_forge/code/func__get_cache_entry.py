import abc
import datetime
from dogpile.cache import api
from dogpile import util as dp_util
from oslo_cache import core
from oslo_log import log
from oslo_utils import importutils
from oslo_utils import timeutils
from oslo_cache._i18n import _
from oslo_cache import exception
def _get_cache_entry(self, key, value, meta, doc_date):
    """MongoDB cache data representation.

        Storing cache key as ``_id`` field as MongoDB by default creates
        unique index on this field. So no need to create separate field and
        index for storing cache key. Cache data has additional ``doc_date``
        field for MongoDB TTL collection support.
        """
    return dict(_id=key, value=value, meta=meta, doc_date=doc_date)
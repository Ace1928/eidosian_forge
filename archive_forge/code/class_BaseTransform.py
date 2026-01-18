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
class BaseTransform(AbstractManipulator):
    """Base transformation class to store and read dogpile cached data
    from MongoDB.

    This is needed as dogpile internally stores data as a custom class
    i.e. dogpile.cache.api.CachedValue

    Note: Custom manipulator needs to always override ``transform_incoming``
    and ``transform_outgoing`` methods. MongoDB manipulator logic specifically
    checks that overridden method in instance and its super are different.
    """

    def transform_incoming(self, son, collection):
        """Used while saving data to MongoDB."""
        for key, value in list(son.items()):
            if isinstance(value, api.CachedValue):
                son[key] = value.payload
                son['meta'] = value.metadata
            elif isinstance(value, dict):
                son[key] = self.transform_incoming(value, collection)
        return son

    def transform_outgoing(self, son, collection):
        """Used while reading data from MongoDB."""
        metadata = None
        if isinstance(son, dict) and all((k in son for k in ('_id', 'value', 'meta', 'doc_date'))):
            payload = son.pop('value', None)
            metadata = son.pop('meta', None)
        for key, value in list(son.items()):
            if isinstance(value, dict):
                son[key] = self.transform_outgoing(value, collection)
        if metadata is not None:
            son['value'] = api.CachedValue(payload, metadata)
        return son
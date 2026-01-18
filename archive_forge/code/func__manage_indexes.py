from __future__ import (absolute_import, division, print_function)
import datetime
from contextlib import contextmanager
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.plugins.cache import BaseCacheModule
from ansible.utils.display import Display
from ansible.module_utils._text import to_native
def _manage_indexes(self, collection):
    """
        This function manages indexes on the mongo collection.
        We only do this once, at run time based on _managed_indexes,
        rather than per connection instantiation as that would be overkill
        """
    _timeout = self._timeout
    if _timeout and _timeout > 0:
        try:
            collection.create_index('date', name='ttl', expireAfterSeconds=_timeout)
        except pymongo.errors.OperationFailure:
            if self._ttl_index_exists(collection):
                collection.drop_index('ttl')
            return self._manage_indexes(collection)
    elif self._ttl_index_exists(collection):
        collection.drop_index('ttl')
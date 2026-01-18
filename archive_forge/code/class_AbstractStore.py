from kivy.clock import Clock
from kivy.event import EventDispatcher
class AbstractStore(EventDispatcher):
    """Abstract class used to implement a Store
    """

    def __init__(self, **kwargs):
        super(AbstractStore, self).__init__(**kwargs)
        self.store_load()

    def exists(self, key):
        """Check if a key exists in the store.
        """
        return self.store_exists(key)

    def async_exists(self, callback, key):
        """Asynchronous version of :meth:`exists`.

        :Callback arguments:
            `store`: :class:`AbstractStore` instance
                Store instance
            `key`: string
                Name of the key to search for
            `result`: boo
                Result of the query, None if any error
        """
        self._schedule(self.store_exists_async, key=key, callback=callback)

    def get(self, key):
        """Get the key-value pairs stored at `key`. If the key is not found, a
        `KeyError` exception will be thrown.
        """
        return self.store_get(key)

    def async_get(self, callback, key):
        """Asynchronous version of :meth:`get`.

        :Callback arguments:
            `store`: :class:`AbstractStore` instance
                Store instance
            `key`: string
                Name of the key to search for
            `result`: dict
                Result of the query, None if any error
        """
        self._schedule(self.store_get_async, key=key, callback=callback)

    def put(self, key, **values):
        """Put new key-value pairs (given in *values*) into the storage. Any
        existing key-value pairs will be removed.
        """
        need_sync = self.store_put(key, values)
        if need_sync:
            self.store_sync()
        return need_sync

    def async_put(self, callback, key, **values):
        """Asynchronous version of :meth:`put`.

        :Callback arguments:
            `store`: :class:`AbstractStore` instance
                Store instance
            `key`: string
                Name of the key to search for
            `result`: bool
                Indicate True if the storage has been updated, or False if
                nothing has been done (no changes). None if any error.
        """
        self._schedule(self.store_put_async, key=key, value=values, callback=callback)

    def delete(self, key):
        """Delete a key from the storage. If the key is not found, a `KeyError`
        exception will be thrown."""
        need_sync = self.store_delete(key)
        if need_sync:
            self.store_sync()
        return need_sync

    def async_delete(self, callback, key):
        """Asynchronous version of :meth:`delete`.

        :Callback arguments:
            `store`: :class:`AbstractStore` instance
                Store instance
            `key`: string
                Name of the key to search for
            `result`: bool
                Indicate True if the storage has been updated, or False if
                nothing has been done (no changes). None if any error.
        """
        self._schedule(self.store_delete_async, key=key, callback=callback)

    def find(self, **filters):
        """Return all the entries matching the filters. The entries are
        returned through a generator as a list of (key, entry) pairs
        where *entry* is a dict of key-value pairs ::

            for key, entry in store.find(name='Mathieu'):
                print('key:', key, ', entry:', entry)

        Because it's a generator, you cannot directly use it as a list. You can
        do::

            # get all the (key, entry) availables
            entries = list(store.find(name='Mathieu'))
            # get only the entry from (key, entry)
            entries = list((x[1] for x in store.find(name='Mathieu')))
        """
        return self.store_find(filters)

    def async_find(self, callback, **filters):
        """Asynchronous version of :meth:`find`.

        The callback will be called for each entry in the result.

        :Callback arguments:
            `store`: :class:`AbstractStore` instance
                Store instance
            `key`: string
                Name of the key to search for, or None if we reach the end of
                the results
            `result`: bool
                Indicate True if the storage has been updated, or False if
                nothing has been done (no changes). None if any error.
        """
        self._schedule(self.store_find_async, callback=callback, filters=filters)

    def keys(self):
        """Return a list of all the keys in the storage.
        """
        return self.store_keys()

    def async_keys(self, callback):
        """Asynchronously return all the keys in the storage.
        """
        self._schedule(self.store_keys_async, callback=callback)

    def count(self):
        """Return the number of entries in the storage.
        """
        return self.store_count()

    def async_count(self, callback):
        """Asynchronously return the number of entries in the storage.
        """
        self._schedule(self.store_count_async, callback=callback)

    def clear(self):
        """Wipe the whole storage.
        """
        return self.store_clear()

    def async_clear(self, callback):
        """Asynchronous version of :meth:`clear`.
        """
        self._schedule(self.store_clear_async, callback=callback)

    def __setitem__(self, key, values):
        if not isinstance(values, dict):
            raise Exception('Only dict are accepted for the store[key] = dict')
        self.put(key, **values)

    def __getitem__(self, key):
        return self.get(key)

    def __delitem__(self, key):
        return self.keys()

    def __contains__(self, key):
        return self.exists(key)

    def __len__(self):
        return self.count()

    def __iter__(self):
        for key in self.keys():
            yield key

    def store_load(self):
        pass

    def store_sync(self):
        pass

    def store_get(self, key):
        raise NotImplementedError

    def store_put(self, key, value):
        raise NotImplementedError

    def store_exists(self, key):
        raise NotImplementedError

    def store_delete(self, key):
        raise NotImplementedError

    def store_find(self, filters):
        return []

    def store_keys(self):
        return []

    def store_count(self):
        return len(self.store_keys())

    def store_clear(self):
        for key in self.store_keys():
            self.store_delete(key)
        self.store_sync()

    def store_get_async(self, key, callback):
        try:
            value = self.store_get(key)
            callback(self, key, value)
        except KeyError:
            callback(self, key, None)

    def store_put_async(self, key, value, callback):
        try:
            value = self.put(key, **value)
            callback(self, key, value)
        except:
            callback(self, key, None)

    def store_exists_async(self, key, callback):
        try:
            value = self.store_exists(key)
            callback(self, key, value)
        except:
            callback(self, key, None)

    def store_delete_async(self, key, callback):
        try:
            value = self.delete(key)
            callback(self, key, value)
        except:
            callback(self, key, None)

    def store_find_async(self, filters, callback):
        for key, entry in self.store_find(filters):
            callback(self, filters, key, entry)
        callback(self, filters, None, None)

    def store_count_async(self, callback):
        try:
            value = self.store_count()
            callback(self, value)
        except:
            callback(self, 0)

    def store_keys_async(self, callback):
        try:
            keys = self.store_keys()
            callback(self, keys)
        except:
            callback(self, [])

    def store_clear_async(self, callback):
        self.store_clear()
        callback(self)

    def _schedule(self, cb, **kwargs):
        Clock.schedule_once(lambda dt: cb(**kwargs), 0)
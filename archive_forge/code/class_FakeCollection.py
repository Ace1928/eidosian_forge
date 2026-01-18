from datetime import datetime
import sys
class FakeCollection(FakeResource):
    """A fake resource for a collection."""

    def __init__(self, application, resource_type, values=None, name=None, child_resource_type=None):
        super(FakeCollection, self).__init__(application, resource_type, values)
        self.__dict__.update({'_name': name, '_child_resource_type': child_resource_type})

    def __iter__(self):
        """Iterate items if this resource has an C{entries} attribute."""
        entries = self._values.get('entries', ())
        for entry in entries:
            yield self._create_resource(self._child_resource_type, self._name, entry)

    def __getitem__(self, key):
        """Look up a slice, or a subordinate resource by index.

        @param key: An individual object key or a C{slice}.
        @raises IndexError: Raised if an invalid key is provided.
        @return: A L{FakeResource} instance for the entry matching C{key}.
        """
        entries = list(self)
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop
            if start < 0:
                raise ValueError('Collection slices must have a nonnegative start point.')
            if stop < 0:
                raise ValueError('Collection slices must have a definite, nonnegative end point.')
            return entries.__getitem__(key)
        elif isinstance(key, int):
            return entries.__getitem__(key)
        else:
            raise IndexError('Do not support index lookups yet.')
from castellan.common.objects import managed_object
class OpaqueData(managed_object.ManagedObject):
    """This class represents opaque data."""

    def __init__(self, data, name=None, created=None, id=None, consumers=[]):
        """Create a new OpaqueData object.

        Expected type for data is a bytestring.
        """
        self._data = data
        super().__init__(name=name, created=created, id=id, consumers=consumers)

    @classmethod
    def managed_type(cls):
        return 'opaque'

    @property
    def format(self):
        return 'Opaque'

    def get_encoded(self):
        return self._data

    def __eq__(self, other):
        if isinstance(other, OpaqueData):
            return self._data == other._data
        else:
            return False

    def __ne__(self, other):
        result = self.__eq__(other)
        return not result
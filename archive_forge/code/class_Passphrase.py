from castellan.common.objects import managed_object
class Passphrase(managed_object.ManagedObject):
    """This class represents a passphrase."""

    def __init__(self, passphrase, name=None, created=None, id=None, consumers=[]):
        """Create a new Passphrase object.

        The expected type for the passphrase is a bytestring.
        """
        self._passphrase = passphrase
        super().__init__(name=name, created=created, id=id, consumers=consumers)

    @classmethod
    def managed_type(cls):
        return 'passphrase'

    @property
    def format(self):
        return 'RAW'

    def get_encoded(self):
        return self._passphrase

    def __eq__(self, other):
        if isinstance(other, Passphrase):
            return self._passphrase == other._passphrase
        else:
            return False

    def __ne__(self, other):
        result = self.__eq__(other)
        return not result
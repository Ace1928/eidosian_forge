from . import errors
class NullIdentityMap:
    """A pretend in memory map from object id to instance.

    A NullIdentityMap is an Identity map that does not store anything in it.
    """

    def add_weave(self, id, weave):
        """See IdentityMap.add_weave."""

    def find_weave(self, id):
        """See IdentityMap.find_weave."""
        return None
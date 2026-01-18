import os
from .... import errors
from .... import transport as _mod_transport
from ....bzr import versionedfile
from ....errors import BzrError, UnlistableStore
from ....trace import mutter
class Store:
    """This class represents the abstract storage layout for saving information.

    Files can be added, but not modified once they are in.  Typically
    the hash is used as the name, or something else known to be unique,
    such as a UUID.
    """

    def __len__(self):
        raise NotImplementedError('Children should define their length')

    def get(self, fileid, suffix=None):
        """Returns a file reading from a particular entry.

        If suffix is present, retrieve the named suffix for fileid.
        """
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def add(self, f, fileid):
        """Add a file object f to the store accessible from the given fileid"""
        raise NotImplementedError('Children of Store must define their method of adding entries.')

    def has_id(self, fileid, suffix=None):
        """Return True or false for the presence of fileid in the store.

        suffix, if present, is a per file suffix, i.e. for digital signature
        data."""
        raise NotImplementedError

    def listable(self):
        """Return True if this store is able to be listed."""
        return getattr(self, '__iter__', None) is not None
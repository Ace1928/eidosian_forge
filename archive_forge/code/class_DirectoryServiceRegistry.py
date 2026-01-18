from typing import Callable, Optional
from . import branch as _mod_branch
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
class DirectoryServiceRegistry(registry.Registry):
    """This object maintains and uses a list of directory services.

    Directory services may be registered via the standard Registry methods.
    They will be invoked if their key is a prefix of the supplied URL.

    Each item registered should be a factory of objects that provide a look_up
    method, as invoked by dereference.  Specifically, look_up should accept a
    name and URL, and return a URL.
    """

    def dereference(self, url, purpose=None):
        """Dereference a supplied URL if possible.

        URLs that match a registered directory service prefix are looked up in
        it.  Non-matching urls are returned verbatim.

        This is applied only once; the resulting URL must not be one that
        requires further dereferencing.

        :param url: The URL to dereference
        :param purpose: Purpose of the URL ('read', 'write' or None - if not declared)
        :return: The dereferenced URL if applicable, the input URL otherwise.
        """
        match = self.get_prefix(url)
        if match is None:
            return url
        service, name = match
        directory = service()
        try:
            return directory.look_up(name, url, purpose=purpose)
        except TypeError:
            return directory.look_up(name, url)
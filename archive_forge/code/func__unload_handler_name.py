import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedTypeError, PasslibWarning
from passlib.ifc import PasswordHash
from passlib.utils import (
from passlib.utils.compat import unicode_or_str
from passlib.utils.decor import memoize_single_value
def _unload_handler_name(name, locations=True):
    """unloads a handler from the registry.

    .. warning::

        this is an internal function,
        used only by the unittests.

    if loaded handler is found with specified name, it's removed.
    if path to lazy load handler is found, it's removed.

    missing names are a noop.

    :arg name: name of handler to unload
    :param locations: if False, won't purge registered handler locations (default True)
    """
    if name in _handlers:
        del _handlers[name]
    if locations and name in _locations:
        del _locations[name]
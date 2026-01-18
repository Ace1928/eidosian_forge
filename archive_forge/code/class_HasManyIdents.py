from __future__ import with_statement
import inspect
import logging; log = logging.getLogger(__name__)
import math
import threading
from warnings import warn
import passlib.exc as exc, passlib.ifc as ifc
from passlib.exc import MissingBackendError, PasslibConfigWarning, \
from passlib.ifc import PasswordHash
from passlib.registry import get_crypt_handler
from passlib.utils import (
from passlib.utils.binary import (
from passlib.utils.compat import join_byte_values, irange, u, native_string_types, \
from passlib.utils.decor import classproperty, deprecated_method
class HasManyIdents(GenericHandler):
    """mixin for hashes which use multiple prefix identifiers

    For the hashes which may use multiple identifier prefixes,
    this mixin adds an ``ident`` keyword to constructor.
    Any value provided is passed through the :meth:`norm_idents` method,
    which takes care of validating the identifier,
    as well as allowing aliases for easier specification
    of the identifiers by the user.

    .. todo::

        document this class's usage

    Class Methods
    =============
    .. todo:: document using() and needs_update() options
    """
    default_ident = None
    ident_values = None
    ident_aliases = None
    ident = None

    @classmethod
    def using(cls, default_ident=None, ident=None, **kwds):
        """
        This mixin adds support for the following :meth:`~passlib.ifc.PasswordHash.using` keywords:

        :param default_ident:
            default identifier that will be used by resulting customized hasher.

        :param ident:
            supported as alternate alias for **default_ident**.
        """
        if ident is not None:
            if default_ident is not None:
                raise TypeError("'default_ident' and 'ident' are mutually exclusive")
            default_ident = ident
        subcls = super(HasManyIdents, cls).using(**kwds)
        if default_ident is not None:
            subcls.default_ident = cls(ident=default_ident, use_defaults=True).ident
        return subcls

    def __init__(self, ident=None, **kwds):
        super(HasManyIdents, self).__init__(**kwds)
        if ident is not None:
            ident = self._norm_ident(ident)
        elif self.use_defaults:
            ident = self.default_ident
            assert validate_default_value(self, ident, self._norm_ident, param='default_ident')
        else:
            raise TypeError('no ident specified')
        self.ident = ident

    @classmethod
    def _norm_ident(cls, ident):
        """
        helper which normalizes & validates 'ident' value.
        """
        assert ident is not None
        if isinstance(ident, bytes):
            ident = ident.decode('ascii')
        iv = cls.ident_values
        if ident in iv:
            return ident
        ia = cls.ident_aliases
        if ia:
            try:
                value = ia[ident]
            except KeyError:
                pass
            else:
                if value in iv:
                    return value
        raise ValueError('invalid ident: %r' % (ident,))

    @classmethod
    def identify(cls, hash):
        hash = to_unicode_for_identify(hash)
        return hash.startswith(cls.ident_values)

    @classmethod
    def _parse_ident(cls, hash):
        """extract ident prefix from hash, helper for subclasses' from_string()"""
        hash = to_unicode(hash, 'ascii', 'hash')
        for ident in cls.ident_values:
            if hash.startswith(ident):
                return (ident, hash[len(ident):])
        raise exc.InvalidHashError(cls)
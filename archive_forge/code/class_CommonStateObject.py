from collections.abc import (
import os
import posixpath
import numpy as np
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode
class CommonStateObject:
    """
        Mixin class that allows sharing information between objects which
        reside in the same HDF5 file.  Requires that the host class have
        a ".id" attribute which returns a low-level ObjectID subclass.

        Also implements Unicode operations.
    """

    @property
    def _lapl(self):
        """ Fetch the link access property list appropriate for this object
        """
        return dlapl

    @property
    def _lcpl(self):
        """ Fetch the link creation property list appropriate for this object
        """
        return dlcpl

    def _e(self, name, lcpl=None):
        """ Encode a name according to the current file settings.

        Returns name, or 2-tuple (name, lcpl) if lcpl is True

        - Binary strings are always passed as-is, h5t.CSET_ASCII
        - Unicode strings are encoded utf8, h5t.CSET_UTF8

        If name is None, returns either None or (None, None) appropriately.
        """

        def get_lcpl(coding):
            """ Create an appropriate link creation property list """
            lcpl = self._lcpl.copy()
            lcpl.set_char_encoding(coding)
            return lcpl
        if name is None:
            return (None, None) if lcpl else None
        if isinstance(name, bytes):
            coding = h5t.CSET_ASCII
        elif isinstance(name, str):
            try:
                name = name.encode('ascii')
                coding = h5t.CSET_ASCII
            except UnicodeEncodeError:
                name = name.encode('utf8')
                coding = h5t.CSET_UTF8
        else:
            raise TypeError(f'A name should be string or bytes, not {type(name)}')
        if lcpl:
            return (name, get_lcpl(coding))
        return name

    def _d(self, name):
        """ Decode a name according to the current file settings.

        - Try to decode utf8
        - Failing that, return the byte string

        If name is None, returns None.
        """
        if name is None:
            return None
        try:
            return name.decode('utf8')
        except UnicodeDecodeError:
            pass
        return name
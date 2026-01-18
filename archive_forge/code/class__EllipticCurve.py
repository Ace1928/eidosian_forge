import calendar
import datetime
import functools
from base64 import b16encode
from functools import partial
from os import PathLike
from typing import (
from cryptography import utils, x509
from cryptography.hazmat.primitives.asymmetric import (
from OpenSSL._util import (
class _EllipticCurve:
    """
    A representation of a supported elliptic curve.

    @cvar _curves: :py:obj:`None` until an attempt is made to load the curves.
        Thereafter, a :py:type:`set` containing :py:type:`_EllipticCurve`
        instances each of which represents one curve supported by the system.
    @type _curves: :py:type:`NoneType` or :py:type:`set`
    """
    _curves = None

    def __ne__(self, other: Any) -> bool:
        """
        Implement cooperation with the right-hand side argument of ``!=``.

        Python 3 seems to have dropped this cooperation in this very narrow
        circumstance.
        """
        if isinstance(other, _EllipticCurve):
            return super(_EllipticCurve, self).__ne__(other)
        return NotImplemented

    @classmethod
    def _load_elliptic_curves(cls, lib: Any) -> Set['_EllipticCurve']:
        """
        Get the curves supported by OpenSSL.

        :param lib: The OpenSSL library binding object.

        :return: A :py:type:`set` of ``cls`` instances giving the names of the
            elliptic curves the underlying library supports.
        """
        num_curves = lib.EC_get_builtin_curves(_ffi.NULL, 0)
        builtin_curves = _ffi.new('EC_builtin_curve[]', num_curves)
        lib.EC_get_builtin_curves(builtin_curves, num_curves)
        return set((cls.from_nid(lib, c.nid) for c in builtin_curves))

    @classmethod
    def _get_elliptic_curves(cls, lib: Any) -> Set['_EllipticCurve']:
        """
        Get, cache, and return the curves supported by OpenSSL.

        :param lib: The OpenSSL library binding object.

        :return: A :py:type:`set` of ``cls`` instances giving the names of the
            elliptic curves the underlying library supports.
        """
        if cls._curves is None:
            cls._curves = cls._load_elliptic_curves(lib)
        return cls._curves

    @classmethod
    def from_nid(cls, lib: Any, nid: int) -> '_EllipticCurve':
        """
        Instantiate a new :py:class:`_EllipticCurve` associated with the given
        OpenSSL NID.

        :param lib: The OpenSSL library binding object.

        :param nid: The OpenSSL NID the resulting curve object will represent.
            This must be a curve NID (and not, for example, a hash NID) or
            subsequent operations will fail in unpredictable ways.
        :type nid: :py:class:`int`

        :return: The curve object.
        """
        return cls(lib, nid, _ffi.string(lib.OBJ_nid2sn(nid)).decode('ascii'))

    def __init__(self, lib: Any, nid: int, name: str) -> None:
        """
        :param _lib: The :py:mod:`cryptography` binding instance used to
            interface with OpenSSL.

        :param _nid: The OpenSSL NID identifying the curve this object
            represents.
        :type _nid: :py:class:`int`

        :param name: The OpenSSL short name identifying the curve this object
            represents.
        :type name: :py:class:`unicode`
        """
        self._lib = lib
        self._nid = nid
        self.name = name

    def __repr__(self) -> str:
        return '<Curve %r>' % (self.name,)

    def _to_EC_KEY(self) -> Any:
        """
        Create a new OpenSSL EC_KEY structure initialized to use this curve.

        The structure is automatically garbage collected when the Python object
        is garbage collected.
        """
        key = self._lib.EC_KEY_new_by_curve_name(self._nid)
        return _ffi.gc(key, _lib.EC_KEY_free)
import sys
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import to_native_str, str_consteq
from passlib.utils.compat import unicode, u, unicode_or_bytes_types
import passlib.utils.handlers as uh
class unix_disabled(uh.ifc.DisabledHash, uh.MinimalHandler):
    """This class provides disabled password behavior for unix shadow files,
    and follows the :ref:`password-hash-api`.

    This class does not implement a hash, but instead matches the "disabled account"
    strings found in ``/etc/shadow`` on most Unix variants. "encrypting" a password
    will simply return the disabled account marker. It will reject all passwords,
    no matter the hash string. The :meth:`~passlib.ifc.PasswordHash.hash`
    method supports one optional keyword:

    :type marker: str
    :param marker:
        Optional marker string which overrides the platform default
        used to indicate a disabled account.

        If not specified, this will default to ``"*"`` on BSD systems,
        and use the Linux default ``"!"`` for all other platforms.
        (:attr:`!unix_disabled.default_marker` will contain the default value)

    .. versionadded:: 1.6
        This class was added as a replacement for the now-deprecated
        :class:`unix_fallback` class, which had some undesirable features.
    """
    name = 'unix_disabled'
    setting_kwds = ('marker',)
    context_kwds = ()
    _disable_prefixes = tuple(str(_MARKER_CHARS))
    if 'bsd' in sys.platform:
        default_marker = u('*')
    else:
        default_marker = u('!')

    @classmethod
    def using(cls, marker=None, **kwds):
        subcls = super(unix_disabled, cls).using(**kwds)
        if marker is not None:
            if not cls.identify(marker):
                raise ValueError('invalid marker: %r' % marker)
            subcls.default_marker = marker
        return subcls

    @classmethod
    def identify(cls, hash):
        if isinstance(hash, unicode):
            start = _MARKER_CHARS
        elif isinstance(hash, bytes):
            start = _MARKER_BYTES
        else:
            raise uh.exc.ExpectedStringError(hash, 'hash')
        return not hash or hash[0] in start

    @classmethod
    def verify(cls, secret, hash):
        uh.validate_secret(secret)
        if not cls.identify(hash):
            raise uh.exc.InvalidHashError(cls)
        return False

    @classmethod
    def hash(cls, secret, **kwds):
        if kwds:
            uh.warn_hash_settings_deprecation(cls, kwds)
            return cls.using(**kwds).hash(secret)
        uh.validate_secret(secret)
        marker = cls.default_marker
        assert marker and cls.identify(marker)
        return to_native_str(marker, param='marker')

    @uh.deprecated_method(deprecated='1.7', removed='2.0')
    @classmethod
    def genhash(cls, secret, config, marker=None):
        if not cls.identify(config):
            raise uh.exc.InvalidHashError(cls)
        elif config:
            uh.validate_secret(secret)
            return to_native_str(config, param='config')
        else:
            if marker is not None:
                cls = cls.using(marker=marker)
            return cls.hash(secret)

    @classmethod
    def disable(cls, hash=None):
        out = cls.hash('')
        if hash is not None:
            hash = to_native_str(hash, param='hash')
            if cls.identify(hash):
                hash = cls.enable(hash)
            if hash:
                out += hash
        return out

    @classmethod
    def enable(cls, hash):
        hash = to_native_str(hash, param='hash')
        for prefix in cls._disable_prefixes:
            if hash.startswith(prefix):
                orig = hash[len(prefix):]
                if orig:
                    return orig
                else:
                    raise ValueError('cannot restore original hash')
        raise uh.exc.InvalidHashError(cls)
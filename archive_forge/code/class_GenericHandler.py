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
class GenericHandler(MinimalHandler):
    """helper class for implementing hash handlers.

    GenericHandler-derived classes will have (at least) the following
    constructor options, though others may be added by mixins
    and by the class itself:

    :param checksum:
        this should contain the digest portion of a
        parsed hash (mainly provided when the constructor is called
        by :meth:`from_string()`).
        defaults to ``None``.

    :param use_defaults:
        If ``False`` (the default), a :exc:`TypeError` should be thrown
        if any settings required by the handler were not explicitly provided.

        If ``True``, the handler should attempt to provide a default for any
        missing values. This means generate missing salts, fill in default
        cost parameters, etc.

        This is typically only set to ``True`` when the constructor
        is called by :meth:`hash`, allowing user-provided values
        to be handled in a more permissive manner.

    :param relaxed:
        If ``False`` (the default), a :exc:`ValueError` should be thrown
        if any settings are out of bounds or otherwise invalid.

        If ``True``, they should be corrected if possible, and a warning
        issue. If not possible, only then should an error be raised.
        (e.g. under ``relaxed=True``, rounds values will be clamped
        to min/max rounds).

        This is mainly used when parsing the config strings of certain
        hashes, whose specifications implementations to be tolerant
        of incorrect values in salt strings.

    Class Attributes
    ================

    .. attribute:: ident

        [optional]
        If this attribute is filled in, the default :meth:`identify` method will use
        it as a identifying prefix that can be used to recognize instances of this handler's
        hash. Filling this out is recommended for speed.

        This should be a unicode str.

    .. attribute:: _hash_regex

        [optional]
        If this attribute is filled in, the default :meth:`identify` method
        will use it to recognize instances of the hash. If :attr:`ident`
        is specified, this will be ignored.

        This should be a unique regex object.

    .. attribute:: checksum_size

        [optional]
        Specifies the number of characters that should be expected in the checksum string.
        If omitted, no check will be performed.

    .. attribute:: checksum_chars

        [optional]
        A string listing all the characters allowed in the checksum string.
        If omitted, no check will be performed.

        This should be a unicode str.

    .. attribute:: _stub_checksum

        Placeholder checksum that will be used by genconfig()
        in lieu of actually generating a hash for the empty string.
        This should be a string of the same datatype as :attr:`checksum`.

    Instance Attributes
    ===================
    .. attribute:: checksum

        The checksum string provided to the constructor (after passing it
        through :meth:`_norm_checksum`).

    Required Subclass Methods
    =========================
    The following methods must be provided by handler subclass:

    .. automethod:: from_string
    .. automethod:: to_string
    .. automethod:: _calc_checksum

    Default Methods
    ===============
    The following methods have default implementations that should work for
    most cases, though they may be overridden if the hash subclass needs to:

    .. automethod:: _norm_checksum

    .. automethod:: genconfig
    .. automethod:: genhash
    .. automethod:: identify
    .. automethod:: hash
    .. automethod:: verify
    """
    setting_kwds = None
    context_kwds = ()
    ident = None
    _hash_regex = None
    checksum_size = None
    checksum_chars = None
    _checksum_is_bytes = False
    checksum = None

    def __init__(self, checksum=None, use_defaults=False, **kwds):
        self.use_defaults = use_defaults
        super(GenericHandler, self).__init__(**kwds)
        if checksum is not None:
            self.checksum = self._norm_checksum(checksum)

    def _norm_checksum(self, checksum, relaxed=False):
        """validates checksum keyword against class requirements,
        returns normalized version of checksum.
        """
        raw = self._checksum_is_bytes
        if raw:
            if not isinstance(checksum, bytes):
                raise exc.ExpectedTypeError(checksum, 'bytes', 'checksum')
        elif not isinstance(checksum, unicode):
            if isinstance(checksum, bytes) and relaxed:
                warn('checksum should be unicode, not bytes', PasslibHashWarning)
                checksum = checksum.decode('ascii')
            else:
                raise exc.ExpectedTypeError(checksum, 'unicode', 'checksum')
        cc = self.checksum_size
        if cc and len(checksum) != cc:
            raise exc.ChecksumSizeError(self, raw=raw)
        if not raw:
            cs = self.checksum_chars
            if cs and any((c not in cs for c in checksum)):
                raise ValueError('invalid characters in %s checksum' % (self.name,))
        return checksum

    @classmethod
    def identify(cls, hash):
        hash = to_unicode_for_identify(hash)
        if not hash:
            return False
        ident = cls.ident
        if ident is not None:
            return hash.startswith(ident)
        pat = cls._hash_regex
        if pat is not None:
            return pat.match(hash) is not None
        try:
            cls.from_string(hash)
            return True
        except ValueError:
            return False

    @classmethod
    def from_string(cls, hash, **context):
        """
        return parsed instance from hash/configuration string

        :param \\\\*\\\\*context:
            context keywords to pass to constructor (if applicable).

        :raises ValueError: if hash is incorrectly formatted

        :returns:
            hash parsed into components,
            for formatting / calculating checksum.
        """
        raise NotImplementedError('%s must implement from_string()' % (cls,))

    def to_string(self):
        """render instance to hash or configuration string

        :returns:
            hash string with salt & digest included.

            should return native string type (ascii-bytes under python 2,
            unicode under python 3)
        """
        raise NotImplementedError('%s must implement from_string()' % (self.__class__,))

    @property
    def _stub_checksum(self):
        """
        placeholder used by default .genconfig() so it can avoid expense of calculating digest.
        """
        if self.checksum_size:
            if self._checksum_is_bytes:
                return b'\x00' * self.checksum_size
            if self.checksum_chars:
                return self.checksum_chars[0] * self.checksum_size
        if isinstance(self, HasRounds):
            orig = self.rounds
            self.rounds = self.min_rounds or 1
            try:
                return self._calc_checksum('')
            finally:
                self.rounds = orig
        return self._calc_checksum('')

    def _calc_checksum(self, secret):
        """given secret; calcuate and return encoded checksum portion of hash
        string, taking config from object state

        calc checksum implementations may assume secret is always
        either unicode or bytes, checks are performed by verify/etc.
        """
        raise NotImplementedError('%s must implement _calc_checksum()' % (self.__class__,))

    @classmethod
    def hash(cls, secret, **kwds):
        if kwds:
            settings = extract_settings_kwds(cls, kwds)
            if settings:
                warn_hash_settings_deprecation(cls, settings)
                return cls.using(**settings).hash(secret, **kwds)
        validate_secret(secret)
        self = cls(use_defaults=True, **kwds)
        self.checksum = self._calc_checksum(secret)
        return self.to_string()

    @classmethod
    def verify(cls, secret, hash, **context):
        validate_secret(secret)
        self = cls.from_string(hash, **context)
        chk = self.checksum
        if chk is None:
            raise exc.MissingDigestError(cls)
        return consteq(self._calc_checksum(secret), chk)

    @deprecated_method(deprecated='1.7', removed='2.0')
    @classmethod
    def genconfig(cls, **kwds):
        settings = extract_settings_kwds(cls, kwds)
        if settings:
            return cls.using(**settings).genconfig(**kwds)
        self = cls(use_defaults=True, **kwds)
        self.checksum = self._stub_checksum
        return self.to_string()

    @deprecated_method(deprecated='1.7', removed='2.0')
    @classmethod
    def genhash(cls, secret, config, **context):
        if config is None:
            raise TypeError('config must be string')
        validate_secret(secret)
        self = cls.from_string(config, **context)
        self.checksum = self._calc_checksum(secret)
        return self.to_string()

    @classmethod
    def needs_update(cls, hash, secret=None, **kwds):
        self = cls.from_string(hash)
        assert isinstance(self, cls)
        return self._calc_needs_update(secret=secret, **kwds)

    def _calc_needs_update(self, secret=None):
        """
        internal helper for :meth:`needs_update`.
        """
        return False
    _always_parse_settings = ()
    _unparsed_settings = ('salt_size', 'relaxed')
    _unsafe_settings = ('salt', 'checksum')

    @classproperty
    def _parsed_settings(cls):
        """
        helper for :meth:`parsehash` --
        returns list of attributes which should be extracted by parse_hash() from hasher object.

        default implementation just takes setting_kwds, and excludes _unparsed_settings
        """
        return tuple((key for key in cls.setting_kwds if key not in cls._unparsed_settings))

    @classmethod
    def parsehash(cls, hash, checksum=True, sanitize=False):
        """[experimental method] parse hash into dictionary of settings.

        this essentially acts as the inverse of :meth:`hash`: for most
        cases, if ``hash = cls.hash(secret, **opts)``, then
        ``cls.parsehash(hash)`` will return a dict matching the original options
        (with the extra keyword *checksum*).

        this method may not work correctly for all hashes,
        and may not be available on some few. its interface may
        change in future releases, if it's kept around at all.

        :arg hash: hash to parse
        :param checksum: include checksum keyword? (defaults to True)
        :param sanitize: mask data for sensitive fields? (defaults to False)
        """
        self = cls.from_string(hash)
        UNSET = object()
        always = self._always_parse_settings
        kwds = dict(((key, getattr(self, key)) for key in self._parsed_settings if key in always or getattr(self, key) != getattr(cls, key, UNSET)))
        if checksum and self.checksum is not None:
            kwds['checksum'] = self.checksum
        if sanitize:
            if sanitize is True:
                sanitize = mask_value
            for key in cls._unsafe_settings:
                if key in kwds:
                    kwds[key] = sanitize(kwds[key])
        return kwds

    @classmethod
    def bitsize(cls, **kwds):
        """[experimental method] return info about bitsizes of hash"""
        try:
            info = super(GenericHandler, cls).bitsize(**kwds)
        except AttributeError:
            info = {}
        cc = ALL_BYTE_VALUES if cls._checksum_is_bytes else cls.checksum_chars
        if cls.checksum_size and cc:
            info['checksum'] = _bitsize(cls.checksum_size, cc)
        return info
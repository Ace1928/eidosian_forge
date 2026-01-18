from __future__ import with_statement, absolute_import
import logging
import re
import types
from warnings import warn
from passlib import exc
from passlib.crypto.digest import MAX_UINT32
from passlib.utils import classproperty, to_bytes, render_bytes
from passlib.utils.binary import b64s_encode, b64s_decode
from passlib.utils.compat import u, unicode, bascii_to_str, uascii_to_str, PY2
import passlib.utils.handlers as uh
class _Argon2Common(uh.SubclassBackendMixin, uh.ParallelismMixin, uh.HasRounds, uh.HasRawSalt, uh.HasRawChecksum, uh.GenericHandler):
    """
    Base class which implements brunt of Argon2 code.
    This is then subclassed by the various backends,
    to override w/ backend-specific methods.

    When a backend is loaded, the bases of the 'argon2' class proper
    are modified to prepend the correct backend-specific subclass.
    """
    name = 'argon2'
    setting_kwds = ('salt', 'salt_size', 'salt_len', 'rounds', 'time_cost', 'memory_cost', 'parallelism', 'digest_size', 'hash_len', 'type')
    checksum_size = _default_settings.hash_len
    _always_parse_settings = uh.GenericHandler._always_parse_settings + ('type',)
    _unparsed_settings = uh.GenericHandler._unparsed_settings + ('salt_len', 'time_cost', 'hash_len', 'digest_size')
    default_salt_size = _default_settings.salt_len
    min_salt_size = 8
    max_salt_size = MAX_UINT32
    default_rounds = _default_settings.time_cost
    min_rounds = 1
    max_rounds = MAX_UINT32
    rounds_cost = 'linear'
    max_parallelism = (1 << 24) - 1
    max_version = _default_version
    min_desired_version = None
    min_memory_cost = 8
    max_threads = -1
    pure_use_threads = False
    _backend_type_map = {}

    @classproperty
    def type_values(cls):
        """
        return tuple of types supported by this backend
        
        .. versionadded:: 1.7.2
        """
        cls.get_backend()
        return tuple(cls._backend_type_map)
    type = TYPE_ID
    parallelism = _default_settings.parallelism
    version = _default_version
    memory_cost = _default_settings.memory_cost

    @property
    def type_d(self):
        """
        flag indicating a Type D hash

        .. deprecated:: 1.7.2; will be removed in passlib 2.0
        """
        return self.type == TYPE_D
    data = None

    @classmethod
    def using(cls, type=None, memory_cost=None, salt_len=None, time_cost=None, digest_size=None, checksum_size=None, hash_len=None, max_threads=None, **kwds):
        if time_cost is not None:
            if 'rounds' in kwds:
                raise TypeError("'time_cost' and 'rounds' are mutually exclusive")
            kwds['rounds'] = time_cost
        if salt_len is not None:
            if 'salt_size' in kwds:
                raise TypeError("'salt_len' and 'salt_size' are mutually exclusive")
            kwds['salt_size'] = salt_len
        if hash_len is not None:
            if digest_size is not None:
                raise TypeError("'hash_len' and 'digest_size' are mutually exclusive")
            digest_size = hash_len
        if checksum_size is not None:
            if digest_size is not None:
                raise TypeError("'checksum_size' and 'digest_size' are mutually exclusive")
            digest_size = checksum_size
        subcls = super(_Argon2Common, cls).using(**kwds)
        if type is not None:
            subcls.type = subcls._norm_type(type)
        relaxed = kwds.get('relaxed')
        if digest_size is not None:
            if isinstance(digest_size, uh.native_string_types):
                digest_size = int(digest_size)
            subcls.checksum_size = uh.norm_integer(subcls, digest_size, min=16, max=MAX_UINT32, param='digest_size', relaxed=relaxed)
        if memory_cost is not None:
            if isinstance(memory_cost, uh.native_string_types):
                memory_cost = int(memory_cost)
            subcls.memory_cost = subcls._norm_memory_cost(memory_cost, relaxed=relaxed)
        subcls._validate_constraints(subcls.memory_cost, subcls.parallelism)
        if max_threads is not None:
            if isinstance(max_threads, uh.native_string_types):
                max_threads = int(max_threads)
            if max_threads < 1 and max_threads != -1:
                raise ValueError('max_threads (%d) must be -1 (unlimited), or at least 1.' % (max_threads,))
            subcls.max_threads = max_threads
        return subcls

    @classmethod
    def _validate_constraints(cls, memory_cost, parallelism):
        min_memory_cost = 8 * parallelism
        if memory_cost < min_memory_cost:
            raise ValueError('%s: memory_cost (%d) is too low, must be at least 8 * parallelism (8 * %d = %d)' % (cls.name, memory_cost, parallelism, min_memory_cost))
    _ident_regex = re.compile('^\\$argon2[a-z]+\\$')

    @classmethod
    def identify(cls, hash):
        hash = uh.to_unicode_for_identify(hash)
        return cls._ident_regex.match(hash) is not None
    _hash_regex = re.compile(b'\n        ^\n        \\$argon2(?P<type>[a-z]+)\\$\n        (?:\n            v=(?P<version>\\d+)\n            \\$\n        )?\n        m=(?P<memory_cost>\\d+)\n        ,\n        t=(?P<time_cost>\\d+)\n        ,\n        p=(?P<parallelism>\\d+)\n        (?:\n            ,keyid=(?P<keyid>[^,$]+)\n        )?\n        (?:\n            ,data=(?P<data>[^,$]+)\n        )?\n        (?:\n            \\$\n            (?P<salt>[^$]+)\n            (?:\n                \\$\n                (?P<digest>.+)\n            )?\n        )?\n        $\n    ', re.X)

    @classmethod
    def from_string(cls, hash):
        if isinstance(hash, unicode):
            hash = hash.encode('utf-8')
        if not isinstance(hash, bytes):
            raise exc.ExpectedStringError(hash, 'hash')
        m = cls._hash_regex.match(hash)
        if not m:
            raise exc.MalformedHashError(cls)
        type, version, memory_cost, time_cost, parallelism, keyid, data, salt, digest = m.group('type', 'version', 'memory_cost', 'time_cost', 'parallelism', 'keyid', 'data', 'salt', 'digest')
        if keyid:
            raise NotImplementedError("argon2 'keyid' parameter not supported")
        return cls(type=type.decode('ascii'), version=int(version) if version else 16, memory_cost=int(memory_cost), rounds=int(time_cost), parallelism=int(parallelism), salt=b64s_decode(salt) if salt else None, data=b64s_decode(data) if data else None, checksum=b64s_decode(digest) if digest else None)

    def to_string(self):
        version = self.version
        if version == 16:
            vstr = ''
        else:
            vstr = 'v=%d$' % version
        data = self.data
        if data:
            kdstr = ',data=' + bascii_to_str(b64s_encode(self.data))
        else:
            kdstr = ''
        return '$argon2%s$%sm=%d,t=%d,p=%d%s$%s$%s' % (uascii_to_str(self.type), vstr, self.memory_cost, self.rounds, self.parallelism, kdstr, bascii_to_str(b64s_encode(self.salt)), bascii_to_str(b64s_encode(self.checksum)))

    def __init__(self, type=None, type_d=False, version=None, memory_cost=None, data=None, **kwds):
        if type_d:
            warn('argon2 `type_d=True` keyword is deprecated, and will be removed in passlib 2.0; please use ``type="d"`` instead')
            assert type is None
            type = TYPE_D
        checksum = kwds.get('checksum')
        if checksum is not None:
            self.checksum_size = len(checksum)
        super(_Argon2Common, self).__init__(**kwds)
        if type is None:
            assert uh.validate_default_value(self, self.type, self._norm_type, param='type')
        else:
            self.type = self._norm_type(type)
        if version is None:
            assert uh.validate_default_value(self, self.version, self._norm_version, param='version')
        else:
            self.version = self._norm_version(version)
        if memory_cost is None:
            assert uh.validate_default_value(self, self.memory_cost, self._norm_memory_cost, param='memory_cost')
        else:
            self.memory_cost = self._norm_memory_cost(memory_cost)
        if data is None:
            assert self.data is None
        else:
            if not isinstance(data, bytes):
                raise uh.exc.ExpectedTypeError(data, 'bytes', 'data')
            self.data = data

    @classmethod
    def _norm_type(cls, value):
        if not isinstance(value, unicode):
            if PY2 and isinstance(value, bytes):
                value = value.decode('ascii')
            else:
                raise uh.exc.ExpectedTypeError(value, 'str', 'type')
        if value in ALL_TYPES_SET:
            return value
        temp = value.lower()
        if temp in ALL_TYPES_SET:
            return temp
        raise ValueError('unknown argon2 hash type: %r' % (value,))

    @classmethod
    def _norm_version(cls, version):
        if not isinstance(version, uh.int_types):
            raise uh.exc.ExpectedTypeError(version, 'integer', 'version')
        if version < 19 and version != 16:
            raise ValueError('invalid argon2 hash version: %d' % (version,))
        backend = cls.get_backend()
        if version > cls.max_version:
            raise ValueError('%s: hash version 0x%X not supported by %r backend (max version is 0x%X); try updating or switching backends' % (cls.name, version, backend, cls.max_version))
        return version

    @classmethod
    def _norm_memory_cost(cls, memory_cost, relaxed=False):
        return uh.norm_integer(cls, memory_cost, min=cls.min_memory_cost, param='memory_cost', relaxed=relaxed)

    @classmethod
    def _get_backend_type(cls, value):
        """
        helper to resolve backend constant from type
        """
        try:
            return cls._backend_type_map[value]
        except KeyError:
            pass
        msg = 'unsupported argon2 hash (type %r not supported by %s backend)' % (value, cls.get_backend())
        raise ValueError(msg)

    def _calc_needs_update(self, **kwds):
        cls = type(self)
        if self.type != cls.type:
            return True
        minver = cls.min_desired_version
        if minver is None or minver > cls.max_version:
            minver = cls.max_version
        if self.version < minver:
            return True
        if self.memory_cost != cls.memory_cost:
            return True
        if self.checksum_size != cls.checksum_size:
            return True
        return super(_Argon2Common, self)._calc_needs_update(**kwds)
    _no_backend_suggestion = " -- recommend you install one (e.g. 'pip install argon2_cffi')"

    @classmethod
    def _finalize_backend_mixin(mixin_cls, name, dryrun):
        """
        helper called by from backend mixin classes' _load_backend_mixin() --
        invoked after backend imports have been loaded, and performs
        feature detection & testing common to all backends.
        """
        max_version = mixin_cls.max_version
        assert isinstance(max_version, int) and max_version >= 16
        if max_version < 19:
            warn("%r doesn't support argon2 v1.3, and should be upgraded" % name, uh.exc.PasslibSecurityWarning)
        for type in ALL_TYPES:
            if type in mixin_cls._backend_type_map:
                mixin_cls.type = type
                break
        else:
            warn('%r lacks support for all known hash types' % name, uh.exc.PasslibRuntimeWarning)
            mixin_cls.type = TYPE_ID
        return True

    @classmethod
    def _adapt_backend_error(cls, err, hash=None, self=None):
        """
        internal helper invoked when backend has hash/verification error;
        used to adapt to passlib message.
        """
        backend = cls.get_backend()
        if self is None and hash is not None:
            self = cls.from_string(hash)
        if self is not None:
            self._validate_constraints(self.memory_cost, self.parallelism)
            if backend == 'argon2_cffi' and self.data is not None:
                raise NotImplementedError("argon2_cffi backend doesn't support the 'data' parameter")
        text = str(err)
        if text not in ['Decoding failed']:
            reason = '%s reported: %s: hash=%r' % (backend, text, hash)
        else:
            reason = repr(hash)
        raise exc.MalformedHashError(cls, reason=reason)
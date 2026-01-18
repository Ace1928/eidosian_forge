from __future__ import with_statement, absolute_import
from base64 import b64encode
from hashlib import sha256
import os
import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.crypto.digest import compile_hmac
from passlib.exc import PasslibHashWarning, PasslibSecurityWarning, PasslibSecurityError
from passlib.utils import safe_crypt, repeat_string, to_bytes, parse_version, \
from passlib.utils.binary import bcrypt64
from passlib.utils.compat import get_unbound_method_function
from passlib.utils.compat import u, uascii_to_str, unicode, str_to_uascii, PY3, error_from
import passlib.utils.handlers as uh
class _BcryptCommon(uh.SubclassBackendMixin, uh.TruncateMixin, uh.HasManyIdents, uh.HasRounds, uh.HasSalt, uh.GenericHandler):
    """
    Base class which implements brunt of BCrypt code.
    This is then subclassed by the various backends,
    to override w/ backend-specific methods.

    When a backend is loaded, the bases of the 'bcrypt' class proper
    are modified to prepend the correct backend-specific subclass.
    """
    name = 'bcrypt'
    setting_kwds = ('salt', 'rounds', 'ident', 'truncate_error')
    checksum_size = 31
    checksum_chars = bcrypt64.charmap
    default_ident = IDENT_2B
    ident_values = (IDENT_2, IDENT_2A, IDENT_2X, IDENT_2Y, IDENT_2B)
    ident_aliases = {u('2'): IDENT_2, u('2a'): IDENT_2A, u('2y'): IDENT_2Y, u('2b'): IDENT_2B}
    min_salt_size = max_salt_size = 22
    salt_chars = bcrypt64.charmap
    final_salt_chars = '.Oeu'
    default_rounds = 12
    min_rounds = 4
    max_rounds = 31
    rounds_cost = 'log2'
    truncate_size = 72
    _workrounds_initialized = False
    _has_2a_wraparound_bug = False
    _lacks_20_support = False
    _lacks_2y_support = False
    _lacks_2b_support = False
    _fallback_ident = IDENT_2A
    _require_valid_utf8_bytes = False

    @classmethod
    def from_string(cls, hash):
        ident, tail = cls._parse_ident(hash)
        if ident == IDENT_2X:
            raise ValueError("crypt_blowfish's buggy '2x' hashes are not currently supported")
        rounds_str, data = tail.split(u('$'))
        rounds = int(rounds_str)
        if rounds_str != u('%02d') % (rounds,):
            raise uh.exc.MalformedHashError(cls, 'malformed cost field')
        salt, chk = (data[:22], data[22:])
        return cls(rounds=rounds, salt=salt, checksum=chk or None, ident=ident)

    def to_string(self):
        hash = u('%s%02d$%s%s') % (self.ident, self.rounds, self.salt, self.checksum)
        return uascii_to_str(hash)

    def _get_config(self, ident):
        """internal helper to prepare config string for backends"""
        config = u('%s%02d$%s') % (ident, self.rounds, self.salt)
        return uascii_to_str(config)

    @classmethod
    def needs_update(cls, hash, **kwds):
        if isinstance(hash, bytes):
            hash = hash.decode('ascii')
        if hash.startswith(IDENT_2A) and hash[28] not in cls.final_salt_chars:
            return True
        return super(_BcryptCommon, cls).needs_update(hash, **kwds)

    @classmethod
    def normhash(cls, hash):
        """helper to normalize hash, correcting any bcrypt padding bits"""
        if cls.identify(hash):
            return cls.from_string(hash).to_string()
        else:
            return hash

    @classmethod
    def _generate_salt(cls):
        salt = super(_BcryptCommon, cls)._generate_salt()
        return bcrypt64.repair_unused(salt)

    @classmethod
    def _norm_salt(cls, salt, **kwds):
        salt = super(_BcryptCommon, cls)._norm_salt(salt, **kwds)
        assert salt is not None, "HasSalt didn't generate new salt!"
        changed, salt = bcrypt64.check_repair_unused(salt)
        if changed:
            warn('encountered a bcrypt salt with incorrectly set padding bits; you may want to use bcrypt.normhash() to fix this; this will be an error under Passlib 2.0', PasslibHashWarning)
        return salt

    def _norm_checksum(self, checksum, relaxed=False):
        checksum = super(_BcryptCommon, self)._norm_checksum(checksum, relaxed=relaxed)
        changed, checksum = bcrypt64.check_repair_unused(checksum)
        if changed:
            warn('encountered a bcrypt hash with incorrectly set padding bits; you may want to use bcrypt.normhash() to fix this; this will be an error under Passlib 2.0', PasslibHashWarning)
        return checksum
    _no_backend_suggestion = " -- recommend you install one (e.g. 'pip install bcrypt')"

    @classmethod
    def _finalize_backend_mixin(mixin_cls, backend, dryrun):
        """
        helper called by from backend mixin classes' _load_backend_mixin() --
        invoked after backend imports have been loaded, and performs
        feature detection & testing common to all backends.
        """
        assert mixin_cls is bcrypt._backend_mixin_map[backend], '_configure_workarounds() invoked from wrong class'
        if mixin_cls._workrounds_initialized:
            return True
        verify = mixin_cls.verify
        err_types = (ValueError, uh.exc.MissingBackendError)
        if _bcryptor:
            err_types += (_bcryptor.engine.SaltError,)

        def safe_verify(secret, hash):
            """verify() wrapper which traps 'unknown identifier' errors"""
            try:
                return verify(secret, hash)
            except err_types:
                return NotImplemented
            except uh.exc.InternalBackendError:
                log.debug('trapped unexpected response from %r backend: verify(%r, %r):', backend, secret, hash, exc_info=True)
                return NotImplemented

        def assert_lacks_8bit_bug(ident):
            """
            helper to check for cryptblowfish 8bit bug (fixed in 2y/2b);
            even though it's not known to be present in any of passlib's backends.
            this is treated as FATAL, because it can easily result in seriously malformed hashes,
            and we can't correct for it ourselves.

            test cases from <http://cvsweb.openwall.com/cgi/cvsweb.cgi/Owl/packages/glibc/crypt_blowfish/wrapper.c.diff?r1=1.9;r2=1.10>
            reference hash is the incorrectly generated $2x$ hash taken from above url
            """
            secret = b'\xd1\x91'
            bug_hash = ident.encode('ascii') + b'05$6bNw2HLQYeqHYyBfLMsv/OiwqTymGIGzFsA4hOTWebfehXHNprcAS'
            correct_hash = ident.encode('ascii') + b'05$6bNw2HLQYeqHYyBfLMsv/OUcZd0LKP39b87nBw3.S2tVZSqiQX6eu'
            if verify(secret, bug_hash):
                raise PasslibSecurityError('passlib.hash.bcrypt: Your installation of the %r backend is vulnerable to the crypt_blowfish 8-bit bug (CVE-2011-2483) under %r hashes, and should be upgraded or replaced with another backend' % (backend, ident))
            if not verify(secret, correct_hash):
                raise RuntimeError('%s backend failed to verify %s 8bit hash' % (backend, ident))

        def detect_wrap_bug(ident):
            """
            check for bsd wraparound bug (fixed in 2b)
            this is treated as a warning, because it's rare in the field,
            and pybcrypt (as of 2015-7-21) is unpatched, but some people may be stuck with it.

            test cases from <http://www.openwall.com/lists/oss-security/2012/01/02/4>

            NOTE: reference hash is of password "0"*72

            NOTE: if in future we need to deliberately create hashes which have this bug,
                  can use something like 'hashpw(repeat_string(secret[:((1+secret) % 256) or 1]), 72)'
            """
            secret = (b'0123456789' * 26)[:255]
            bug_hash = ident.encode('ascii') + b'04$R1lJ2gkNaoPGdafE.H.16.nVyh2niHsGJhayOHLMiXlI45o8/DU.6'
            if verify(secret, bug_hash):
                return True
            correct_hash = ident.encode('ascii') + b'04$R1lJ2gkNaoPGdafE.H.16.1MKHPvmKwryeulRe225LKProWYwt9Oi'
            if not verify(secret, correct_hash):
                raise RuntimeError('%s backend failed to verify %s wraparound hash' % (backend, ident))
            return False

        def assert_lacks_wrap_bug(ident):
            if not detect_wrap_bug(ident):
                return
            raise RuntimeError('%s backend unexpectedly has wraparound bug for %s' % (backend, ident))
        test_hash_20 = b'$2$04$5BJqKfqMQvV7nS.yUguNcuRfMMOXK0xPWavM7pOzjEi5ze5T1k8/S'
        result = safe_verify('test', test_hash_20)
        if result is NotImplemented:
            mixin_cls._lacks_20_support = True
            log.debug('%r backend lacks $2$ support, enabling workaround', backend)
        elif not result:
            raise RuntimeError('%s incorrectly rejected $2$ hash' % backend)
        result = safe_verify('test', TEST_HASH_2A)
        if result is NotImplemented:
            raise RuntimeError('%s lacks support for $2a$ hashes' % backend)
        elif not result:
            raise RuntimeError('%s incorrectly rejected $2a$ hash' % backend)
        else:
            assert_lacks_8bit_bug(IDENT_2A)
            if detect_wrap_bug(IDENT_2A):
                if backend == 'os_crypt':
                    log.debug('%r backend has $2a$ bsd wraparound bug, enabling workaround', backend)
                else:
                    warn('passlib.hash.bcrypt: Your installation of the %r backend is vulnerable to the bsd wraparound bug, and should be upgraded or replaced with another backend (enabling workaround for now).' % backend, uh.exc.PasslibSecurityWarning)
                mixin_cls._has_2a_wraparound_bug = True
        test_hash_2y = TEST_HASH_2A.replace('2a', '2y')
        result = safe_verify('test', test_hash_2y)
        if result is NotImplemented:
            mixin_cls._lacks_2y_support = True
            log.debug('%r backend lacks $2y$ support, enabling workaround', backend)
        elif not result:
            raise RuntimeError('%s incorrectly rejected $2y$ hash' % backend)
        else:
            assert_lacks_8bit_bug(IDENT_2Y)
            assert_lacks_wrap_bug(IDENT_2Y)
        test_hash_2b = TEST_HASH_2A.replace('2a', '2b')
        result = safe_verify('test', test_hash_2b)
        if result is NotImplemented:
            mixin_cls._lacks_2b_support = True
            log.debug('%r backend lacks $2b$ support, enabling workaround', backend)
        elif not result:
            raise RuntimeError('%s incorrectly rejected $2b$ hash' % backend)
        else:
            mixin_cls._fallback_ident = IDENT_2B
            assert_lacks_8bit_bug(IDENT_2B)
            assert_lacks_wrap_bug(IDENT_2B)
        mixin_cls._workrounds_initialized = True
        return True

    def _prepare_digest_args(self, secret):
        """
        common helper for backends to implement _calc_checksum().
        takes in secret, returns (secret, ident) pair,
        """
        return self._norm_digest_args(secret, self.ident, new=self.use_defaults)

    @classmethod
    def _norm_digest_args(cls, secret, ident, new=False):
        require_valid_utf8_bytes = cls._require_valid_utf8_bytes
        if isinstance(secret, unicode):
            secret = secret.encode('utf-8')
        elif require_valid_utf8_bytes:
            try:
                secret.decode('utf-8')
            except UnicodeDecodeError:
                require_valid_utf8_bytes = False
        uh.validate_secret(secret)
        if new:
            cls._check_truncate_policy(secret)
        if _BNULL in secret:
            raise uh.exc.NullPasswordError(cls)
        if cls._has_2a_wraparound_bug and len(secret) >= 255:
            if require_valid_utf8_bytes:
                secret = utf8_truncate(secret, 72)
            else:
                secret = secret[:72]
        if ident == IDENT_2A:
            pass
        elif ident == IDENT_2B:
            if cls._lacks_2b_support:
                ident = cls._fallback_ident
        elif ident == IDENT_2Y:
            if cls._lacks_2y_support:
                ident = cls._fallback_ident
        elif ident == IDENT_2:
            if cls._lacks_20_support:
                if secret:
                    if require_valid_utf8_bytes:
                        secret = utf8_repeat_string(secret, 72)
                    else:
                        secret = repeat_string(secret, 72)
                ident = cls._fallback_ident
        elif ident == IDENT_2X:
            raise RuntimeError('$2x$ hashes not currently supported by passlib')
        else:
            raise AssertionError('unexpected ident value: %r' % ident)
        return (secret, ident)
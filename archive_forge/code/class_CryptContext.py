from __future__ import with_statement
import re
import logging; log = logging.getLogger(__name__)
import threading
import time
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedStringError, ExpectedTypeError, PasslibConfigWarning
from passlib.registry import get_crypt_handler, _validate_handler_name
from passlib.utils import (handlers as uh, to_bytes,
from passlib.utils.binary import BASE64_CHARS
from passlib.utils.compat import (iteritems, num_types, irange,
from passlib.utils.decor import deprecated_method, memoized_property
class CryptContext(object):
    """Helper for hashing & verifying passwords using multiple algorithms.

    Instances of this class allow applications to choose a specific
    set of hash algorithms which they wish to support, set limits and defaults
    for the rounds and salt sizes those algorithms should use, flag
    which algorithms should be deprecated, and automatically handle
    migrating users to stronger hashes when they log in.

    Basic usage::

        >>> ctx = CryptContext(schemes=[...])

    See the Passlib online documentation for details and full documentation.
    """
    _config = None
    _get_record = None
    _identify_record = None

    @classmethod
    def _norm_source(cls, source):
        """internal helper - accepts string, dict, or context"""
        if isinstance(source, dict):
            return cls(**source)
        elif isinstance(source, cls):
            return source
        else:
            self = cls()
            self.load(source)
            return self

    @classmethod
    def from_string(cls, source, section='passlib', encoding='utf-8'):
        """create new CryptContext instance from an INI-formatted string.

        :type source: unicode or bytes
        :arg source:
            string containing INI-formatted content.

        :type section: str
        :param section:
            option name of section to read from, defaults to ``"passlib"``.

        :type encoding: str
        :arg encoding:
            optional encoding used when source is bytes, defaults to ``"utf-8"``.

        :returns:
            new :class:`CryptContext` instance, configured based on the
            parameters in the *source* string.

        Usage example::

            >>> from passlib.context import CryptContext
            >>> context = CryptContext.from_string('''
            ... [passlib]
            ... schemes = sha256_crypt, des_crypt
            ... sha256_crypt__default_rounds = 30000
            ... ''')

        .. versionadded:: 1.6

        .. seealso:: :meth:`to_string`, the inverse of this constructor.
        """
        if not isinstance(source, unicode_or_bytes_types):
            raise ExpectedTypeError(source, 'unicode or bytes', 'source')
        self = cls(_autoload=False)
        self.load(source, section=section, encoding=encoding)
        return self

    @classmethod
    def from_path(cls, path, section='passlib', encoding='utf-8'):
        """create new CryptContext instance from an INI-formatted file.

        this functions exactly the same as :meth:`from_string`,
        except that it loads from a local file.

        :type path: str
        :arg path:
            path to local file containing INI-formatted config.

        :type section: str
        :param section:
            option name of section to read from, defaults to ``"passlib"``.

        :type encoding: str
        :arg encoding:
            encoding used to load file, defaults to ``"utf-8"``.

        :returns:
            new CryptContext instance, configured based on the parameters
            stored in the file *path*.

        .. versionadded:: 1.6

        .. seealso:: :meth:`from_string` for an equivalent usage example.
        """
        self = cls(_autoload=False)
        self.load_path(path, section=section, encoding=encoding)
        return self

    def copy(self, **kwds):
        """Return copy of existing CryptContext instance.

        This function returns a new CryptContext instance whose configuration
        is exactly the same as the original, with the exception that any keywords
        passed in will take precedence over the original settings.
        As an example::

            >>> from passlib.context import CryptContext

            >>> # given an existing context...
            >>> ctx1 = CryptContext(["sha256_crypt", "md5_crypt"])

            >>> # copy can be used to make a clone, and update
            >>> # some of the settings at the same time...
            >>> ctx2 = custom_app_context.copy(default="md5_crypt")

            >>> # and the original will be unaffected by the change
            >>> ctx1.default_scheme()
            "sha256_crypt"
            >>> ctx2.default_scheme()
            "md5_crypt"

        .. versionadded:: 1.6
            This method was previously named :meth:`!replace`. That alias
            has been deprecated, and will be removed in Passlib 1.8.

        .. seealso:: :meth:`update`
        """
        other = CryptContext(_autoload=False)
        other.load(self)
        if kwds:
            other.load(kwds, update=True)
        return other

    def using(self, **kwds):
        """
        alias for :meth:`copy`, to match PasswordHash.using()
        """
        return self.copy(**kwds)

    def replace(self, **kwds):
        """deprecated alias of :meth:`copy`"""
        warn('CryptContext().replace() has been deprecated in Passlib 1.6, and will be removed in Passlib 1.8, it has been renamed to CryptContext().copy()', DeprecationWarning, stacklevel=2)
        return self.copy(**kwds)

    def __init__(self, schemes=None, policy=_UNSET, _autoload=True, **kwds):
        if schemes is not None:
            kwds['schemes'] = schemes
        if policy is not _UNSET:
            warn('The CryptContext ``policy`` keyword has been deprecated as of Passlib 1.6, and will be removed in Passlib 1.8; please use ``CryptContext.from_string()` or ``CryptContext.from_path()`` instead.', DeprecationWarning)
            if policy is None:
                self.load(kwds)
            elif isinstance(policy, CryptPolicy):
                self.load(policy._context)
                self.update(kwds)
            else:
                raise TypeError('policy must be a CryptPolicy instance')
        elif _autoload:
            self.load(kwds)
        else:
            assert not kwds, '_autoload=False and kwds are mutually exclusive'

    def __repr__(self):
        return '<CryptContext at 0x%0x>' % id(self)

    def _get_policy(self):
        return CryptPolicy(_internal_context=self.copy(), _stub_policy=True)

    def _set_policy(self, policy):
        warn('The CryptPolicy class and the ``context.policy`` attribute have been deprecated as of Passlib 1.6, and will be removed in Passlib 1.8; please use the ``context.load()`` and ``context.update()`` methods instead.', DeprecationWarning, stacklevel=2)
        if isinstance(policy, CryptPolicy):
            self.load(policy._context)
        else:
            raise TypeError('expected CryptPolicy instance')
    policy = property(_get_policy, _set_policy, doc='[deprecated] returns CryptPolicy instance tied to this CryptContext')

    @staticmethod
    def _parse_ini_stream(stream, section, filename):
        """helper read INI from stream, extract passlib section as dict"""
        p = SafeConfigParser()
        if PY3:
            p.read_file(stream, filename)
        else:
            p.readfp(stream, filename)
        return dict(p.items(section))

    def load_path(self, path, update=False, section='passlib', encoding='utf-8'):
        """Load new configuration into CryptContext from a local file.

        This function is a wrapper for :meth:`load` which
        loads a configuration string from the local file *path*,
        instead of an in-memory source. Its behavior and options
        are otherwise identical to :meth:`!load` when provided with
        an INI-formatted string.

        .. versionadded:: 1.6
        """

        def helper(stream):
            kwds = self._parse_ini_stream(stream, section, path)
            return self.load(kwds, update=update)
        if PY3:
            with open(path, 'rt', encoding=encoding) as stream:
                return helper(stream)
        elif encoding in ['utf-8', 'ascii']:
            with open(path, 'rb') as stream:
                return helper(stream)
        else:
            with open(path, 'rb') as fh:
                tmp = fh.read().decode(encoding).encode('utf-8')
                return helper(BytesIO(tmp))

    def load(self, source, update=False, section='passlib', encoding='utf-8'):
        """Load new configuration into CryptContext, replacing existing config.

        :arg source:
            source of new configuration to load.
            this value can be a number of different types:

            * a :class:`!dict` object, or compatible Mapping

                the key/value pairs will be interpreted the same
                keywords for the :class:`CryptContext` class constructor.

            * a :class:`!unicode` or :class:`!bytes` string

                this will be interpreted as an INI-formatted file,
                and appropriate key/value pairs will be loaded from
                the specified *section*.

            * another :class:`!CryptContext` object.

                this will export a snapshot of its configuration
                using :meth:`to_dict`.

        :type update: bool
        :param update:
            By default, :meth:`load` will replace the existing configuration
            entirely. If ``update=True``, it will preserve any existing
            configuration options that are not overridden by the new source,
            much like the :meth:`update` method.

        :type section: str
        :param section:
            When parsing an INI-formatted string, :meth:`load` will look for
            a section named ``"passlib"``. This option allows an alternate
            section name to be used. Ignored when loading from a dictionary.

        :type encoding: str
        :param encoding:
            Encoding to use when **source** is bytes.
            Defaults to ``"utf-8"``. Ignored when loading from a dictionary.

            .. deprecated:: 1.8

                This keyword, and support for bytes input, will be dropped in Passlib 2.0

        :raises TypeError:
            * If the source cannot be identified.
            * If an unknown / malformed keyword is encountered.

        :raises ValueError:
            If an invalid keyword value is encountered.

        .. note::

            If an error occurs during a :meth:`!load` call, the :class:`!CryptContext`
            instance will be restored to the configuration it was in before
            the :meth:`!load` call was made; this is to ensure it is
            *never* left in an inconsistent state due to a load error.

        .. versionadded:: 1.6
        """
        parse_keys = True
        if isinstance(source, unicode_or_bytes_types):
            if PY3:
                source = to_unicode(source, encoding, param='source')
            else:
                source = to_bytes(source, 'utf-8', source_encoding=encoding, param='source')
            source = self._parse_ini_stream(NativeStringIO(source), section, '<string passed to CryptContext.load()>')
        elif isinstance(source, CryptContext):
            source = dict(source._config.iter_config(resolve=True))
            parse_keys = False
        elif not hasattr(source, 'items'):
            raise ExpectedTypeError(source, 'string or dict', 'source')
        if parse_keys:
            parse = self._parse_config_key
            source = dict(((parse(key), value) for key, value in iteritems(source)))
        if update and self._config is not None:
            if not source:
                return
            tmp = source
            source = dict(self._config.iter_config(resolve=True))
            source.update(tmp)
        config = _CryptConfig(source)
        self._config = config
        self._reset_dummy_verify()
        self._get_record = config.get_record
        self._identify_record = config.identify_record
        if config.context_kwds:
            self.__dict__.pop('_strip_unused_context_kwds', None)
        else:
            self._strip_unused_context_kwds = None

    @staticmethod
    def _parse_config_key(ckey):
        """helper used to parse ``cat__scheme__option`` keys into a tuple"""
        assert isinstance(ckey, native_string_types)
        parts = ckey.replace('.', '__').split('__')
        count = len(parts)
        if count == 1:
            cat, scheme, key = (None, None, parts[0])
        elif count == 2:
            cat = None
            scheme, key = parts
        elif count == 3:
            cat, scheme, key = parts
        else:
            raise TypeError('keys must have less than 3 separators: %r' % (ckey,))
        if cat == 'default':
            cat = None
        elif not cat and cat is not None:
            raise TypeError('empty category: %r' % ckey)
        if scheme == 'context':
            scheme = None
        elif not scheme and scheme is not None:
            raise TypeError('empty scheme: %r' % ckey)
        if not key:
            raise TypeError('empty option: %r' % ckey)
        return (cat, scheme, key)

    def update(self, *args, **kwds):
        """Helper for quickly changing configuration.

        This acts much like the :meth:`!dict.update` method:
        it updates the context's configuration,
        replacing the original value(s) for the specified keys,
        and preserving the rest.
        It accepts any :ref:`keyword <context-options>`
        accepted by the :class:`!CryptContext` constructor.

        .. versionadded:: 1.6

        .. seealso:: :meth:`copy`
        """
        if args:
            if len(args) > 1:
                raise TypeError('expected at most one positional argument')
            if kwds:
                raise TypeError('positional arg and keywords mutually exclusive')
            self.load(args[0], update=True)
        elif kwds:
            self.load(kwds, update=True)

    def schemes(self, resolve=False, category=None, unconfigured=False):
        """return schemes loaded into this CryptContext instance.

        :type resolve: bool
        :arg resolve:
            if ``True``, will return a tuple of :class:`~passlib.ifc.PasswordHash`
            objects instead of their names.

        :returns:
            returns tuple of the schemes configured for this context
            via the *schemes* option.

        .. versionadded:: 1.6
            This was previously available as ``CryptContext().policy.schemes()``

        .. seealso:: the :ref:`schemes <context-schemes-option>` option for usage example.
        """
        schemes = self._config.schemes
        if resolve:
            return tuple((self.handler(scheme, category, unconfigured=unconfigured) for scheme in schemes))
        else:
            return schemes

    def default_scheme(self, category=None, resolve=False, unconfigured=False):
        """return name of scheme that :meth:`hash` will use by default.

        :type resolve: bool
        :arg resolve:
            if ``True``, will return a :class:`~passlib.ifc.PasswordHash`
            object instead of the name.

        :type category: str or None
        :param category:
            Optional :ref:`user category <user-categories>`.
            If specified, this will return the catgory-specific default scheme instead.

        :returns:
            name of the default scheme.

        .. seealso:: the :ref:`default <context-default-option>` option for usage example.

        .. versionadded:: 1.6

        .. versionchanged:: 1.7

            This now returns a hasher configured with any CryptContext-specific
            options (custom rounds settings, etc).  Previously this returned
            the base hasher from :mod:`passlib.hash`.
        """
        hasher = self.handler(None, category, unconfigured=unconfigured)
        return hasher if resolve else hasher.name

    def handler(self, scheme=None, category=None, unconfigured=False):
        """helper to resolve name of scheme -> :class:`~passlib.ifc.PasswordHash` object used by scheme.

        :arg scheme:
            This should identify the scheme to lookup.
            If omitted or set to ``None``, this will return the handler
            for the default scheme.

        :arg category:
            If a user category is specified, and no scheme is provided,
            it will use the default for that category.
            Otherwise this parameter is ignored.

        :param unconfigured:

            By default, this returns a handler object whose .hash()
            and .needs_update() methods will honor the configured
            provided by CryptContext.   See ``unconfigured=True``
            to get the underlying handler from before any context-specific
            configuration was applied.

        :raises KeyError:
            If the scheme does not exist OR is not being used within this context.

        :returns:
            :class:`~passlib.ifc.PasswordHash` object used to implement
            the named scheme within this context (this will usually
            be one of the objects from :mod:`passlib.hash`)

        .. versionadded:: 1.6
            This was previously available as ``CryptContext().policy.get_handler()``

        .. versionchanged:: 1.7

            This now returns a hasher configured with any CryptContext-specific
            options (custom rounds settings, etc).  Previously this returned
            the base hasher from :mod:`passlib.hash`.
        """
        try:
            hasher = self._get_record(scheme, category)
            if unconfigured:
                return hasher._Context__orig_handler
            else:
                return hasher
        except KeyError:
            pass
        if self._config.handlers:
            raise KeyError('crypt algorithm not found in this CryptContext instance: %r' % (scheme,))
        else:
            raise KeyError('no crypt algorithms loaded in this CryptContext instance')

    def _get_unregistered_handlers(self):
        """check if any handlers in this context aren't in the global registry"""
        return tuple((handler for handler in self._config.handlers if not _is_handler_registered(handler)))

    @property
    def context_kwds(self):
        """
        return :class:`!set` containing union of all :ref:`contextual keywords <context-keywords>`
        supported by the handlers in this context.

        .. versionadded:: 1.6.6
        """
        return self._config.context_kwds

    @staticmethod
    def _render_config_key(key):
        """convert 3-part config key to single string"""
        cat, scheme, option = key
        if cat:
            return '%s__%s__%s' % (cat, scheme or 'context', option)
        elif scheme:
            return '%s__%s' % (scheme, option)
        else:
            return option

    @staticmethod
    def _render_ini_value(key, value):
        """render value to string suitable for INI file"""
        if isinstance(value, (list, tuple)):
            value = ', '.join(value)
        elif isinstance(value, num_types):
            if isinstance(value, float) and key[2] == 'vary_rounds':
                value = ('%.2f' % value).rstrip('0') if value else '0'
            else:
                value = str(value)
        assert isinstance(value, native_string_types), 'expected string for key: %r %r' % (key, value)
        return value.replace('%', '%%')

    def to_dict(self, resolve=False):
        """Return current configuration as a dictionary.

        :type resolve: bool
        :arg resolve:
            if ``True``, the ``schemes`` key will contain a list of
            a :class:`~passlib.ifc.PasswordHash` objects instead of just
            their names.

        This method dumps the current configuration of the CryptContext
        instance. The key/value pairs should be in the format accepted
        by the :class:`!CryptContext` class constructor, in fact
        ``CryptContext(**myctx.to_dict())`` will create an exact copy of ``myctx``.
        As an example::

            >>> # you can dump the configuration of any crypt context...
            >>> from passlib.apps import ldap_nocrypt_context
            >>> ldap_nocrypt_context.to_dict()
            {'schemes': ['ldap_salted_sha1',
            'ldap_salted_md5',
            'ldap_sha1',
            'ldap_md5',
            'ldap_plaintext']}

        .. versionadded:: 1.6
            This was previously available as ``CryptContext().policy.to_dict()``

        .. seealso:: the :ref:`context-serialization-example` example in the tutorial.
        """
        render_key = self._render_config_key
        return dict(((render_key(key), value) for key, value in self._config.iter_config(resolve)))

    def _write_to_parser(self, parser, section):
        """helper to write to ConfigParser instance"""
        render_key = self._render_config_key
        render_value = self._render_ini_value
        parser.add_section(section)
        for k, v in self._config.iter_config():
            v = render_value(k, v)
            k = render_key(k)
            parser.set(section, k, v)

    def to_string(self, section='passlib'):
        """serialize to INI format and return as unicode string.

        :param section:
            name of INI section to output, defaults to ``"passlib"``.

        :returns:
            CryptContext configuration, serialized to a INI unicode string.

        This function acts exactly like :meth:`to_dict`, except that it
        serializes all the contents into a single human-readable string,
        which can be hand edited, and/or stored in a file. The
        output of this method is accepted by :meth:`from_string`,
        :meth:`from_path`, and :meth:`load`. As an example::

            >>> # you can dump the configuration of any crypt context...
            >>> from passlib.apps import ldap_nocrypt_context
            >>> print ldap_nocrypt_context.to_string()
            [passlib]
            schemes = ldap_salted_sha1, ldap_salted_md5, ldap_sha1, ldap_md5, ldap_plaintext

        .. versionadded:: 1.6
            This was previously available as ``CryptContext().policy.to_string()``

        .. seealso:: the :ref:`context-serialization-example` example in the tutorial.
        """
        parser = SafeConfigParser()
        self._write_to_parser(parser, section)
        buf = NativeStringIO()
        parser.write(buf)
        unregistered = self._get_unregistered_handlers()
        if unregistered:
            buf.write('# NOTE: the %s handler(s) are not registered with Passlib,\n# this string may not correctly reproduce the current configuration.\n\n' % ', '.join((repr(handler.name) for handler in unregistered)))
        out = buf.getvalue()
        if not PY3:
            out = out.decode('utf-8')
        return out
    mvt_estimate_max_samples = 20
    mvt_estimate_min_samples = 10
    mvt_estimate_max_time = 2
    mvt_estimate_resolution = 0.01
    harden_verify = None
    min_verify_time = 0

    def reset_min_verify_time(self):
        self._reset_dummy_verify()

    def _get_or_identify_record(self, hash, scheme=None, category=None):
        """return record based on scheme, or failing that, by identifying hash"""
        if scheme:
            if not isinstance(hash, unicode_or_bytes_types):
                raise ExpectedStringError(hash, 'hash')
            return self._get_record(scheme, category)
        else:
            return self._identify_record(hash, category)

    def _strip_unused_context_kwds(self, kwds, record):
        """
        helper which removes any context keywords from **kwds**
        that are known to be used by another scheme in this context,
        but are NOT supported by handler specified by **record**.

        .. note::
            as optimization, load() will set this method to None on a per-instance basis
            if there are no context kwds.
        """
        if not kwds:
            return
        unused_kwds = self._config.context_kwds.difference(record.context_kwds)
        for key in unused_kwds:
            kwds.pop(key, None)

    def needs_update(self, hash, scheme=None, category=None, secret=None):
        """Check if hash needs to be replaced for some reason,
        in which case the secret should be re-hashed.

        This function is the core of CryptContext's support for hash migration:
        This function takes in a hash string, and checks the scheme,
        number of rounds, and other properties against the current policy.
        It returns ``True`` if the hash is using a deprecated scheme,
        or is otherwise outside of the bounds specified by the policy
        (e.g. the number of rounds is lower than :ref:`min_rounds <context-min-rounds-option>`
        configuration for that algorithm).
        If so, the password should be re-hashed using :meth:`hash`
        Otherwise, it will return ``False``.

        :type hash: unicode or bytes
        :arg hash:
            The hash string to examine.

        :type scheme: str or None
        :param scheme:

            Optional scheme to use. Scheme must be one of the ones
            configured for this context (see the
            :ref:`schemes <context-schemes-option>` option).
            If no scheme is specified, it will be identified
            based on the value of *hash*.

            .. deprecated:: 1.7

                Support for this keyword is deprecated, and will be removed in Passlib 2.0.

        :type category: str or None
        :param category:
            Optional :ref:`user category <user-categories>`.
            If specified, this will cause any category-specific defaults to
            be used when determining if the hash needs to be updated
            (e.g. is below the minimum rounds).

        :type secret: unicode, bytes, or None
        :param secret:
            Optional secret associated with the provided ``hash``.
            This is not required, or even currently used for anything...
            it's for forward-compatibility with any future
            update checks that might need this information.
            If provided, Passlib assumes the secret has already been
            verified successfully against the hash.

            .. versionadded:: 1.6

        :returns: ``True`` if hash should be replaced, otherwise ``False``.

        :raises ValueError:
            If the hash did not match any of the configured :meth:`schemes`.

        .. versionadded:: 1.6
            This method was previously named :meth:`hash_needs_update`.

        .. seealso:: the :ref:`context-migration-example` example in the tutorial.
        """
        if scheme is not None:
            warn("CryptContext.needs_update(): 'scheme' keyword is deprecated as of Passlib 1.7, and will be removed in Passlib 2.0", DeprecationWarning)
        record = self._get_or_identify_record(hash, scheme, category)
        return record.deprecated or record.needs_update(hash, secret=secret)

    @deprecated_method(deprecated='1.6', removed='2.0', replacement='CryptContext.needs_update()')
    def hash_needs_update(self, hash, scheme=None, category=None):
        """Legacy alias for :meth:`needs_update`.

        .. deprecated:: 1.6
            This method was renamed to :meth:`!needs_update` in version 1.6.
            This alias will be removed in version 2.0, and should only
            be used for compatibility with Passlib 1.3 - 1.5.
        """
        return self.needs_update(hash, scheme, category)

    @deprecated_method(deprecated='1.7', removed='2.0')
    def genconfig(self, scheme=None, category=None, **settings):
        """Generate a config string for specified scheme.

        .. deprecated:: 1.7

            This method will be removed in version 2.0, and should only
            be used for compatibility with Passlib 1.3 - 1.6.
        """
        record = self._get_record(scheme, category)
        strip_unused = self._strip_unused_context_kwds
        if strip_unused:
            strip_unused(settings, record)
        return record.genconfig(**settings)

    @deprecated_method(deprecated='1.7', removed='2.0')
    def genhash(self, secret, config, scheme=None, category=None, **kwds):
        """Generate hash for the specified secret using another hash.

        .. deprecated:: 1.7

            This method will be removed in version 2.0, and should only
            be used for compatibility with Passlib 1.3 - 1.6.
        """
        record = self._get_or_identify_record(config, scheme, category)
        strip_unused = self._strip_unused_context_kwds
        if strip_unused:
            strip_unused(kwds, record)
        return record.genhash(secret, config, **kwds)

    def identify(self, hash, category=None, resolve=False, required=False, unconfigured=False):
        """Attempt to identify which algorithm the hash belongs to.

        Note that this will only consider the algorithms
        currently configured for this context
        (see the :ref:`schemes <context-schemes-option>` option).
        All registered algorithms will be checked, from first to last,
        and whichever one positively identifies the hash first will be returned.

        :type hash: unicode or bytes
        :arg hash:
            The hash string to test.

        :type category: str or None
        :param category:
            Optional :ref:`user category <user-categories>`.
            Ignored by this function, this parameter
            is provided for symmetry with the other methods.

        :type resolve: bool
        :param resolve:
            If ``True``, returns the hash handler itself,
            instead of the name of the hash.

        :type required: bool
        :param required:
            If ``True``, this will raise a ValueError if the hash
            cannot be identified, instead of returning ``None``.

        :returns:
            The handler which first identifies the hash,
            or ``None`` if none of the algorithms identify the hash.
        """
        record = self._identify_record(hash, category, required)
        if record is None:
            return None
        elif resolve:
            if unconfigured:
                return record._Context__orig_handler
            else:
                return record
        else:
            return record.name

    def hash(self, secret, scheme=None, category=None, **kwds):
        """run secret through selected algorithm, returning resulting hash.

        :type secret: unicode or bytes
        :arg secret:
            the password to hash.

        :type scheme: str or None
        :param scheme:

            Optional scheme to use. Scheme must be one of the ones
            configured for this context (see the
            :ref:`schemes <context-schemes-option>` option).
            If no scheme is specified, the configured default
            will be used.

            .. deprecated:: 1.7

                Support for this keyword is deprecated, and will be removed in Passlib 2.0.

        :type category: str or None
        :param category:
            Optional :ref:`user category <user-categories>`.
            If specified, this will cause any category-specific defaults to
            be used when hashing the password (e.g. different default scheme,
            different default rounds values, etc).

        :param \\*\\*kwds:
            All other keyword options are passed to the selected algorithm's
            :meth:`PasswordHash.hash() <passlib.ifc.PasswordHash.hash>` method.

        :returns:
            The secret as encoded by the specified algorithm and options.
            The return value will always be a :class:`!str`.

        :raises TypeError, ValueError:
            * If any of the arguments have an invalid type or value.
              This includes any keywords passed to the underlying hash's
              :meth:`PasswordHash.hash() <passlib.ifc.PasswordHash.hash>` method.

        .. seealso:: the :ref:`context-basic-example` example in the tutorial
        """
        if scheme is not None:
            warn("CryptContext.hash(): 'scheme' keyword is deprecated as of Passlib 1.7, and will be removed in Passlib 2.0", DeprecationWarning)
        record = self._get_record(scheme, category)
        strip_unused = self._strip_unused_context_kwds
        if strip_unused:
            strip_unused(kwds, record)
        return record.hash(secret, **kwds)

    @deprecated_method(deprecated='1.7', removed='2.0', replacement='CryptContext.hash()')
    def encrypt(self, *args, **kwds):
        """
        Legacy alias for :meth:`hash`.

        .. deprecated:: 1.7
            This method was renamed to :meth:`!hash` in version 1.7.
            This alias will be removed in version 2.0, and should only
            be used for compatibility with Passlib 1.3 - 1.6.
        """
        return self.hash(*args, **kwds)

    def verify(self, secret, hash, scheme=None, category=None, **kwds):
        """verify secret against an existing hash.

        If no scheme is specified, this will attempt to identify
        the scheme based on the contents of the provided hash
        (limited to the schemes configured for this context).
        It will then check whether the password verifies against the hash.

        :type secret: unicode or bytes
        :arg secret:
            the secret to verify

        :type hash: unicode or bytes
        :arg hash:
            hash string to compare to

            if ``None`` is passed in, this will be treated as "never verifying"

        :type scheme: str
        :param scheme:
            Optionally force context to use specific scheme.
            This is usually not needed, as most hashes can be unambiguously
            identified. Scheme must be one of the ones configured
            for this context
            (see the :ref:`schemes <context-schemes-option>` option).

            .. deprecated:: 1.7

                Support for this keyword is deprecated, and will be removed in Passlib 2.0.

        :type category: str or None
        :param category:
            Optional :ref:`user category <user-categories>` string.
            This is mainly used when generating new hashes, it has little
            effect when verifying; this keyword is mainly provided for symmetry.

        :param \\*\\*kwds:
            All additional keywords are passed to the appropriate handler,
            and should match its :attr:`~passlib.ifc.PasswordHash.context_kwds`.

        :returns:
            ``True`` if the password matched the hash, else ``False``.

        :raises ValueError:
            * if the hash did not match any of the configured :meth:`schemes`.

            * if any of the arguments have an invalid value (this includes
              any keywords passed to the underlying hash's
              :meth:`PasswordHash.verify() <passlib.ifc.PasswordHash.verify>` method).

        :raises TypeError:
            * if any of the arguments have an invalid type (this includes
              any keywords passed to the underlying hash's
              :meth:`PasswordHash.verify() <passlib.ifc.PasswordHash.verify>` method).

        .. seealso:: the :ref:`context-basic-example` example in the tutorial
        """
        if scheme is not None:
            warn("CryptContext.verify(): 'scheme' keyword is deprecated as of Passlib 1.7, and will be removed in Passlib 2.0", DeprecationWarning)
        if hash is None:
            self.dummy_verify()
            return False
        record = self._get_or_identify_record(hash, scheme, category)
        strip_unused = self._strip_unused_context_kwds
        if strip_unused:
            strip_unused(kwds, record)
        return record.verify(secret, hash, **kwds)

    def verify_and_update(self, secret, hash, scheme=None, category=None, **kwds):
        """verify password and re-hash the password if needed, all in a single call.

        This is a convenience method which takes care of all the following:
        first it verifies the password (:meth:`~CryptContext.verify`), if this is successfull
        it checks if the hash needs updating (:meth:`~CryptContext.needs_update`), and if so,
        re-hashes the password (:meth:`~CryptContext.hash`), returning the replacement hash.
        This series of steps is a very common task for applications
        which wish to update deprecated hashes, and this call takes
        care of all 3 steps efficiently.

        :type secret: unicode or bytes
        :arg secret:
            the secret to verify

        :type secret: unicode or bytes
        :arg hash:
            hash string to compare to.

            if ``None`` is passed in, this will be treated as "never verifying"

        :type scheme: str
        :param scheme:
            Optionally force context to use specific scheme.
            This is usually not needed, as most hashes can be unambiguously
            identified. Scheme must be one of the ones configured
            for this context
            (see the :ref:`schemes <context-schemes-option>` option).

            .. deprecated:: 1.7

                Support for this keyword is deprecated, and will be removed in Passlib 2.0.

        :type category: str or None
        :param category:
            Optional :ref:`user category <user-categories>`.
            If specified, this will cause any category-specific defaults to
            be used if the password has to be re-hashed.

        :param \\*\\*kwds:
            all additional keywords are passed to the appropriate handler,
            and should match that hash's
            :attr:`PasswordHash.context_kwds <passlib.ifc.PasswordHash.context_kwds>`.

        :returns:
            This function returns a tuple containing two elements:
            ``(verified, replacement_hash)``. The first is a boolean
            flag indicating whether the password verified,
            and the second an optional replacement hash.
            The tuple will always match one of the following 3 cases:

            * ``(False, None)`` indicates the secret failed to verify.
            * ``(True, None)`` indicates the secret verified correctly,
              and the hash does not need updating.
            * ``(True, str)`` indicates the secret verified correctly,
              but the current hash needs to be updated. The :class:`!str`
              will be the freshly generated hash, to replace the old one.

        :raises TypeError, ValueError:
            For the same reasons as :meth:`verify`.

        .. seealso:: the :ref:`context-migration-example` example in the tutorial.
        """
        if scheme is not None:
            warn("CryptContext.verify(): 'scheme' keyword is deprecated as of Passlib 1.7, and will be removed in Passlib 2.0", DeprecationWarning)
        if hash is None:
            self.dummy_verify()
            return (False, None)
        record = self._get_or_identify_record(hash, scheme, category)
        strip_unused = self._strip_unused_context_kwds
        if strip_unused and kwds:
            clean_kwds = kwds.copy()
            strip_unused(clean_kwds, record)
        else:
            clean_kwds = kwds
        if not record.verify(secret, hash, **clean_kwds):
            return (False, None)
        elif record.deprecated or record.needs_update(hash, secret=secret):
            return (True, self.hash(secret, category=category, **kwds))
        else:
            return (True, None)
    _dummy_secret = 'too many secrets'

    @memoized_property
    def _dummy_hash(self):
        """
        precalculated hash for dummy_verify() to use
        """
        return self.hash(self._dummy_secret)

    def _reset_dummy_verify(self):
        """
        flush memoized values used by dummy_verify()
        """
        type(self)._dummy_hash.clear_cache(self)

    def dummy_verify(self, elapsed=0):
        """
        Helper that applications can call when user wasn't found,
        in order to simulate time it would take to hash a password.

        Runs verify() against a dummy hash, to simulate verification
        of a real account password.

        :param elapsed:

            .. deprecated:: 1.7.1

                this option is ignored, and will be removed in passlib 1.8.

        .. versionadded:: 1.7
        """
        self.verify(self._dummy_secret, self._dummy_hash)
        return False

    def is_enabled(self, hash):
        """
        test if hash represents a usuable password --
        i.e. does not represent an unusuable password such as ``"!"``,
        which is recognized by the :class:`~passlib.hash.unix_disabled` hash.

        :raises ValueError:
            if the hash is not recognized
            (typically solved by adding ``unix_disabled`` to the list of schemes).
        """
        return not self._identify_record(hash, None).is_disabled

    def disable(self, hash=None):
        """
        return a string to disable logins for user,
        usually by returning a non-verifying string such as ``"!"``.

        :param hash:
            Callers can optionally provide the account's existing hash.
            Some disabled handlers (such as :class:`!unix_disabled`)
            will encode this into the returned value,
            so that it can be recovered via :meth:`enable`.

        :raises RuntimeError:
            if this function is called w/o a disabled hasher
            (such as :class:`~passlib.hash.unix_disabled`) included
            in the list of schemes.

        :returns:
            hash string which will be recognized as valid by the context,
            but is guaranteed to not validate against *any* password.
        """
        record = self._config.disabled_record
        assert record.is_disabled
        return record.disable(hash)

    def enable(self, hash):
        """
        inverse of :meth:`disable` --
        attempts to recover original hash which was converted
        by a :meth:`!disable` call into a disabled hash --
        thus restoring the user's original password.

        :raises ValueError:
            if original hash not present, or if the disabled handler doesn't
            support encoding the original hash (e.g. ``django_disabled``)

        :returns:
            the original hash.
        """
        record = self._identify_record(hash, None)
        if record.is_disabled:
            return record.enable(hash)
        else:
            return hash
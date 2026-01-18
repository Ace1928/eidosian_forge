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
class CryptPolicy(object):
    """
    .. deprecated:: 1.6
        This class has been deprecated, and will be removed in Passlib 1.8.
        All of its functionality has been rolled into :class:`CryptContext`.

    This class previously stored the configuration options for the
    CryptContext class. In the interest of interface simplification,
    all of this class' functionality has been rolled into the CryptContext
    class itself.
    The documentation for this class is now focused on  documenting how to
    migrate to the new api. Additionally, where possible, the deprecation
    warnings issued by the CryptPolicy methods will list the replacement call
    that should be used.

    Constructors
    ============
    CryptPolicy objects can be constructed directly using any of
    the keywords accepted by :class:`CryptContext`. Direct uses of the
    :class:`!CryptPolicy` constructor should either pass the keywords
    directly into the CryptContext constructor, or to :meth:`CryptContext.update`
    if the policy object was being used to update an existing context object.

    In addition to passing in keywords directly,
    CryptPolicy objects can be constructed by the following methods:

    .. automethod:: from_path
    .. automethod:: from_string
    .. automethod:: from_source
    .. automethod:: from_sources
    .. automethod:: replace

    Introspection
    =============
    All of the informational methods provided by this class have been deprecated
    by identical or similar methods in the :class:`CryptContext` class:

    .. automethod:: has_schemes
    .. automethod:: schemes
    .. automethod:: iter_handlers
    .. automethod:: get_handler
    .. automethod:: get_options
    .. automethod:: handler_is_deprecated
    .. automethod:: get_min_verify_time

    Exporting
    =========
    .. automethod:: iter_config
    .. automethod:: to_dict
    .. automethod:: to_file
    .. automethod:: to_string

    .. note::
        CryptPolicy are immutable.
        Use the :meth:`replace` method to mutate existing instances.

    .. deprecated:: 1.6
    """

    @classmethod
    def from_path(cls, path, section='passlib', encoding='utf-8'):
        """create a CryptPolicy instance from a local file.

        .. deprecated:: 1.6

        Creating a new CryptContext from a file, which was previously done via
        ``CryptContext(policy=CryptPolicy.from_path(path))``, can now be
        done via ``CryptContext.from_path(path)``.
        See :meth:`CryptContext.from_path` for details.

        Updating an existing CryptContext from a file, which was previously done
        ``context.policy = CryptPolicy.from_path(path)``, can now be
        done via ``context.load_path(path)``.
        See :meth:`CryptContext.load_path` for details.
        """
        warn(_preamble + 'Instead of ``CryptPolicy.from_path(path)``, use ``CryptContext.from_path(path)``  or ``context.load_path(path)`` for an existing CryptContext.', DeprecationWarning, stacklevel=2)
        return cls(_internal_context=CryptContext.from_path(path, section, encoding))

    @classmethod
    def from_string(cls, source, section='passlib', encoding='utf-8'):
        """create a CryptPolicy instance from a string.

        .. deprecated:: 1.6

        Creating a new CryptContext from a string, which was previously done via
        ``CryptContext(policy=CryptPolicy.from_string(data))``, can now be
        done via ``CryptContext.from_string(data)``.
        See :meth:`CryptContext.from_string` for details.

        Updating an existing CryptContext from a string, which was previously done
        ``context.policy = CryptPolicy.from_string(data)``, can now be
        done via ``context.load(data)``.
        See :meth:`CryptContext.load` for details.
        """
        warn(_preamble + 'Instead of ``CryptPolicy.from_string(source)``, use ``CryptContext.from_string(source)`` or ``context.load(source)`` for an existing CryptContext.', DeprecationWarning, stacklevel=2)
        return cls(_internal_context=CryptContext.from_string(source, section, encoding))

    @classmethod
    def from_source(cls, source, _warn=True):
        """create a CryptPolicy instance from some source.

        this method autodetects the source type, and invokes
        the appropriate constructor automatically. it attempts
        to detect whether the source is a configuration string, a filepath,
        a dictionary, or an existing CryptPolicy instance.

        .. deprecated:: 1.6

        Create a new CryptContext, which could previously be done via
        ``CryptContext(policy=CryptPolicy.from_source(source))``, should
        now be done using an explicit method: the :class:`CryptContext`
        constructor itself, :meth:`CryptContext.from_path`,
        or :meth:`CryptContext.from_string`.

        Updating an existing CryptContext, which could previously be done via
        ``context.policy = CryptPolicy.from_source(source)``, should
        now be done using an explicit method: :meth:`CryptContext.update`,
        or :meth:`CryptContext.load`.
        """
        if _warn:
            warn(_preamble + 'Instead of ``CryptPolicy.from_source()``, use ``CryptContext.from_string(path)``  or ``CryptContext.from_path(source)``, as appropriate.', DeprecationWarning, stacklevel=2)
        if isinstance(source, CryptPolicy):
            return source
        elif isinstance(source, dict):
            return cls(_internal_context=CryptContext(**source))
        elif not isinstance(source, (bytes, unicode)):
            raise TypeError('source must be CryptPolicy, dict, config string, or file path: %r' % (type(source),))
        elif any((c in source for c in '\n\r\t')) or not source.strip(' \t./;:'):
            return cls(_internal_context=CryptContext.from_string(source))
        else:
            return cls(_internal_context=CryptContext.from_path(source))

    @classmethod
    def from_sources(cls, sources, _warn=True):
        """create a CryptPolicy instance by merging multiple sources.

        each source is interpreted as by :meth:`from_source`,
        and the results are merged together.

        .. deprecated:: 1.6
            Instead of using this method to merge multiple policies together,
            a :class:`CryptContext` instance should be created, and then
            the multiple sources merged together via :meth:`CryptContext.load`.
        """
        if _warn:
            warn(_preamble + 'Instead of ``CryptPolicy.from_sources()``, use the various CryptContext constructors  followed by ``context.update()``.', DeprecationWarning, stacklevel=2)
        if len(sources) == 0:
            raise ValueError('no sources specified')
        if len(sources) == 1:
            return cls.from_source(sources[0], _warn=False)
        kwds = {}
        for source in sources:
            kwds.update(cls.from_source(source, _warn=False)._context.to_dict(resolve=True))
        return cls(_internal_context=CryptContext(**kwds))

    def replace(self, *args, **kwds):
        """create a new CryptPolicy, optionally updating parts of the
        existing configuration.

        .. deprecated:: 1.6
            Callers of this method should :meth:`CryptContext.update` or
            :meth:`CryptContext.copy` instead.
        """
        if self._stub_policy:
            warn(_preamble + 'Instead of ``context.policy.replace()``, use ``context.update()`` or ``context.copy()``.', DeprecationWarning, stacklevel=2)
        else:
            warn(_preamble + 'Instead of ``CryptPolicy().replace()``, create a CryptContext instance and use ``context.update()`` or ``context.copy()``.', DeprecationWarning, stacklevel=2)
        sources = [self]
        if args:
            sources.extend(args)
        if kwds:
            sources.append(kwds)
        return CryptPolicy.from_sources(sources, _warn=False)
    _context = None
    _stub_policy = False

    def __init__(self, *args, **kwds):
        context = kwds.pop('_internal_context', None)
        if context:
            assert isinstance(context, CryptContext)
            self._context = context
            self._stub_policy = kwds.pop('_stub_policy', False)
            assert not (args or kwds), 'unexpected args: %r %r' % (args, kwds)
        else:
            if args:
                if len(args) != 1:
                    raise TypeError('only one positional argument accepted')
                if kwds:
                    raise TypeError('cannot specify positional arg and kwds')
                kwds = args[0]
            warn(_preamble + 'Instead of constructing a CryptPolicy instance, create a CryptContext directly, or use ``context.update()`` and ``context.load()`` to reconfigure existing CryptContext instances.', DeprecationWarning, stacklevel=2)
            self._context = CryptContext(**kwds)

    def has_schemes(self):
        """return True if policy defines *any* schemes for use.

        .. deprecated:: 1.6
            applications should use ``bool(context.schemes())`` instead.
            see :meth:`CryptContext.schemes`.
        """
        if self._stub_policy:
            warn(_preamble + 'Instead of ``context.policy.has_schemes()``, use ``bool(context.schemes())``.', DeprecationWarning, stacklevel=2)
        else:
            warn(_preamble + 'Instead of ``CryptPolicy().has_schemes()``, create a CryptContext instance and use ``bool(context.schemes())``.', DeprecationWarning, stacklevel=2)
        return bool(self._context.schemes())

    def iter_handlers(self):
        """return iterator over handlers defined in policy.

        .. deprecated:: 1.6
            applications should use ``context.schemes(resolve=True))`` instead.
            see :meth:`CryptContext.schemes`.
        """
        if self._stub_policy:
            warn(_preamble + 'Instead of ``context.policy.iter_handlers()``, use ``context.schemes(resolve=True)``.', DeprecationWarning, stacklevel=2)
        else:
            warn(_preamble + 'Instead of ``CryptPolicy().iter_handlers()``, create a CryptContext instance and use ``context.schemes(resolve=True)``.', DeprecationWarning, stacklevel=2)
        return self._context.schemes(resolve=True, unconfigured=True)

    def schemes(self, resolve=False):
        """return list of schemes defined in policy.

        .. deprecated:: 1.6
            applications should use :meth:`CryptContext.schemes` instead.
        """
        if self._stub_policy:
            warn(_preamble + 'Instead of ``context.policy.schemes()``, use ``context.schemes()``.', DeprecationWarning, stacklevel=2)
        else:
            warn(_preamble + 'Instead of ``CryptPolicy().schemes()``, create a CryptContext instance and use ``context.schemes()``.', DeprecationWarning, stacklevel=2)
        return list(self._context.schemes(resolve=resolve, unconfigured=True))

    def get_handler(self, name=None, category=None, required=False):
        """return handler as specified by name, or default handler.

        .. deprecated:: 1.6
            applications should use :meth:`CryptContext.handler` instead,
            though note that the ``required`` keyword has been removed,
            and the new method will always act as if ``required=True``.
        """
        if self._stub_policy:
            warn(_preamble + 'Instead of ``context.policy.get_handler()``, use ``context.handler()``.', DeprecationWarning, stacklevel=2)
        else:
            warn(_preamble + 'Instead of ``CryptPolicy().get_handler()``, create a CryptContext instance and use ``context.handler()``.', DeprecationWarning, stacklevel=2)
        try:
            return self._context.handler(name, category, unconfigured=True)
        except KeyError:
            if required:
                raise
            else:
                return None

    def get_min_verify_time(self, category=None):
        """get min_verify_time setting for policy.

        .. deprecated:: 1.6
            min_verify_time option will be removed entirely in passlib 1.8

        .. versionchanged:: 1.7
            this method now always returns the value automatically
            calculated by :meth:`CryptContext.min_verify_time`,
            any value specified by policy is ignored.
        """
        warn('get_min_verify_time() and min_verify_time option is deprecated and ignored, and will be removed in Passlib 1.8', DeprecationWarning, stacklevel=2)
        return 0

    def get_options(self, name, category=None):
        """return dictionary of options specific to a given handler.

        .. deprecated:: 1.6
            this method has no direct replacement in the 1.6 api, as there
            is not a clearly defined use-case. however, examining the output of
            :meth:`CryptContext.to_dict` should serve as the closest alternative.
        """
        if self._stub_policy:
            warn(_preamble + '``context.policy.get_options()`` will no longer be available.', DeprecationWarning, stacklevel=2)
        else:
            warn(_preamble + '``CryptPolicy().get_options()`` will no longer be available.', DeprecationWarning, stacklevel=2)
        if hasattr(name, 'name'):
            name = name.name
        return self._context._config._get_record_options_with_flag(name, category)[0]

    def handler_is_deprecated(self, name, category=None):
        """check if handler has been deprecated by policy.

        .. deprecated:: 1.6
            this method has no direct replacement in the 1.6 api, as there
            is not a clearly defined use-case. however, examining the output of
            :meth:`CryptContext.to_dict` should serve as the closest alternative.
        """
        if self._stub_policy:
            warn(_preamble + '``context.policy.handler_is_deprecated()`` will no longer be available.', DeprecationWarning, stacklevel=2)
        else:
            warn(_preamble + '``CryptPolicy().handler_is_deprecated()`` will no longer be available.', DeprecationWarning, stacklevel=2)
        if hasattr(name, 'name'):
            name = name.name
        return self._context.handler(name, category).deprecated

    def iter_config(self, ini=False, resolve=False):
        """iterate over key/value pairs representing the policy object.

        .. deprecated:: 1.6
            applications should use :meth:`CryptContext.to_dict` instead.
        """
        if self._stub_policy:
            warn(_preamble + 'Instead of ``context.policy.iter_config()``, use ``context.to_dict().items()``.', DeprecationWarning, stacklevel=2)
        else:
            warn(_preamble + 'Instead of ``CryptPolicy().iter_config()``, create a CryptContext instance and use ``context.to_dict().items()``.', DeprecationWarning, stacklevel=2)
        context = self._context
        if ini:

            def render_key(key):
                return context._render_config_key(key).replace('__', '.')

            def render_value(value):
                if isinstance(value, (list, tuple)):
                    value = ', '.join(value)
                return value
            resolve = False
        else:
            render_key = context._render_config_key
            render_value = lambda value: value
        return ((render_key(key), render_value(value)) for key, value in context._config.iter_config(resolve))

    def to_dict(self, resolve=False):
        """export policy object as dictionary of options.

        .. deprecated:: 1.6
            applications should use :meth:`CryptContext.to_dict` instead.
        """
        if self._stub_policy:
            warn(_preamble + 'Instead of ``context.policy.to_dict()``, use ``context.to_dict()``.', DeprecationWarning, stacklevel=2)
        else:
            warn(_preamble + 'Instead of ``CryptPolicy().to_dict()``, create a CryptContext instance and use ``context.to_dict()``.', DeprecationWarning, stacklevel=2)
        return self._context.to_dict(resolve)

    def to_file(self, stream, section='passlib'):
        """export policy to file.

        .. deprecated:: 1.6
            applications should use :meth:`CryptContext.to_string` instead,
            and then write the output to a file as desired.
        """
        if self._stub_policy:
            warn(_preamble + 'Instead of ``context.policy.to_file(stream)``, use ``stream.write(context.to_string())``.', DeprecationWarning, stacklevel=2)
        else:
            warn(_preamble + 'Instead of ``CryptPolicy().to_file(stream)``, create a CryptContext instance and use ``stream.write(context.to_string())``.', DeprecationWarning, stacklevel=2)
        out = self._context.to_string(section=section)
        if PY2:
            out = out.encode('utf-8')
        stream.write(out)

    def to_string(self, section='passlib', encoding=None):
        """export policy to file.

        .. deprecated:: 1.6
            applications should use :meth:`CryptContext.to_string` instead.
        """
        if self._stub_policy:
            warn(_preamble + 'Instead of ``context.policy.to_string()``, use ``context.to_string()``.', DeprecationWarning, stacklevel=2)
        else:
            warn(_preamble + 'Instead of ``CryptPolicy().to_string()``, create a CryptContext instance and use ``context.to_string()``.', DeprecationWarning, stacklevel=2)
        out = self._context.to_string(section=section)
        if encoding:
            out = out.encode(encoding)
        return out
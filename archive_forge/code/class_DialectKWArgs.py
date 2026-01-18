from __future__ import annotations
import collections
from enum import Enum
import itertools
from itertools import zip_longest
import operator
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import roles
from . import visitors
from .cache_key import HasCacheKey  # noqa
from .cache_key import MemoizedHasCacheKey  # noqa
from .traversals import HasCopyInternals  # noqa
from .visitors import ClauseVisitor
from .visitors import ExtendedInternalTraversal
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import util
from ..util import HasMemoized as HasMemoized
from ..util import hybridmethod
from ..util import typing as compat_typing
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeGuard
class DialectKWArgs:
    """Establish the ability for a class to have dialect-specific arguments
    with defaults and constructor validation.

    The :class:`.DialectKWArgs` interacts with the
    :attr:`.DefaultDialect.construct_arguments` present on a dialect.

    .. seealso::

        :attr:`.DefaultDialect.construct_arguments`

    """
    __slots__ = ()
    _dialect_kwargs_traverse_internals = [('dialect_options', InternalTraversal.dp_dialect_options)]

    @classmethod
    def argument_for(cls, dialect_name, argument_name, default):
        """Add a new kind of dialect-specific keyword argument for this class.

        E.g.::

            Index.argument_for("mydialect", "length", None)

            some_index = Index('a', 'b', mydialect_length=5)

        The :meth:`.DialectKWArgs.argument_for` method is a per-argument
        way adding extra arguments to the
        :attr:`.DefaultDialect.construct_arguments` dictionary. This
        dictionary provides a list of argument names accepted by various
        schema-level constructs on behalf of a dialect.

        New dialects should typically specify this dictionary all at once as a
        data member of the dialect class.  The use case for ad-hoc addition of
        argument names is typically for end-user code that is also using
        a custom compilation scheme which consumes the additional arguments.

        :param dialect_name: name of a dialect.  The dialect must be
         locatable, else a :class:`.NoSuchModuleError` is raised.   The
         dialect must also include an existing
         :attr:`.DefaultDialect.construct_arguments` collection, indicating
         that it participates in the keyword-argument validation and default
         system, else :class:`.ArgumentError` is raised.  If the dialect does
         not include this collection, then any keyword argument can be
         specified on behalf of this dialect already.  All dialects packaged
         within SQLAlchemy include this collection, however for third party
         dialects, support may vary.

        :param argument_name: name of the parameter.

        :param default: default value of the parameter.

        """
        construct_arg_dictionary = DialectKWArgs._kw_registry[dialect_name]
        if construct_arg_dictionary is None:
            raise exc.ArgumentError("Dialect '%s' does have keyword-argument validation and defaults enabled configured" % dialect_name)
        if cls not in construct_arg_dictionary:
            construct_arg_dictionary[cls] = {}
        construct_arg_dictionary[cls][argument_name] = default

    @util.memoized_property
    def dialect_kwargs(self):
        """A collection of keyword arguments specified as dialect-specific
        options to this construct.

        The arguments are present here in their original ``<dialect>_<kwarg>``
        format.  Only arguments that were actually passed are included;
        unlike the :attr:`.DialectKWArgs.dialect_options` collection, which
        contains all options known by this dialect including defaults.

        The collection is also writable; keys are accepted of the
        form ``<dialect>_<kwarg>`` where the value will be assembled
        into the list of options.

        .. seealso::

            :attr:`.DialectKWArgs.dialect_options` - nested dictionary form

        """
        return _DialectArgView(self)

    @property
    def kwargs(self):
        """A synonym for :attr:`.DialectKWArgs.dialect_kwargs`."""
        return self.dialect_kwargs
    _kw_registry = util.PopulateDict(_kw_reg_for_dialect)

    def _kw_reg_for_dialect_cls(self, dialect_name):
        construct_arg_dictionary = DialectKWArgs._kw_registry[dialect_name]
        d = _DialectArgDict()
        if construct_arg_dictionary is None:
            d._defaults.update({'*': None})
        else:
            for cls in reversed(self.__class__.__mro__):
                if cls in construct_arg_dictionary:
                    d._defaults.update(construct_arg_dictionary[cls])
        return d

    @util.memoized_property
    def dialect_options(self):
        """A collection of keyword arguments specified as dialect-specific
        options to this construct.

        This is a two-level nested registry, keyed to ``<dialect_name>``
        and ``<argument_name>``.  For example, the ``postgresql_where``
        argument would be locatable as::

            arg = my_object.dialect_options['postgresql']['where']

        .. versionadded:: 0.9.2

        .. seealso::

            :attr:`.DialectKWArgs.dialect_kwargs` - flat dictionary form

        """
        return util.PopulateDict(util.portable_instancemethod(self._kw_reg_for_dialect_cls))

    def _validate_dialect_kwargs(self, kwargs: Dict[str, Any]) -> None:
        if not kwargs:
            return
        for k in kwargs:
            m = re.match('^(.+?)_(.+)$', k)
            if not m:
                raise TypeError("Additional arguments should be named <dialectname>_<argument>, got '%s'" % k)
            dialect_name, arg_name = m.group(1, 2)
            try:
                construct_arg_dictionary = self.dialect_options[dialect_name]
            except exc.NoSuchModuleError:
                util.warn("Can't validate argument %r; can't locate any SQLAlchemy dialect named %r" % (k, dialect_name))
                self.dialect_options[dialect_name] = d = _DialectArgDict()
                d._defaults.update({'*': None})
                d._non_defaults[arg_name] = kwargs[k]
            else:
                if '*' not in construct_arg_dictionary and arg_name not in construct_arg_dictionary:
                    raise exc.ArgumentError('Argument %r is not accepted by dialect %r on behalf of %r' % (k, dialect_name, self.__class__))
                else:
                    construct_arg_dictionary[arg_name] = kwargs[k]
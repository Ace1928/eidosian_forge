import sys
import typing
from struct import Struct
from types import TracebackType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union
from docutils import nodes
from docutils.parsers.rst.states import Inliner
from sphinx.deprecation import RemovedInSphinx60Warning, deprecated_alias
def _restify_py37(cls: Optional[Type], mode: str='fully-qualified-except-typing') -> str:
    """Convert python class to a reST reference."""
    from sphinx.util import inspect
    if mode == 'smart':
        modprefix = '~'
    else:
        modprefix = ''
    if inspect.isgenericalias(cls) and cls.__module__ == 'typing' and (cls.__origin__ is Union):
        if len(cls.__args__) > 1 and cls.__args__[-1] is NoneType:
            if len(cls.__args__) > 2:
                args = ', '.join((restify(a, mode) for a in cls.__args__[:-1]))
                return ':py:obj:`~typing.Optional`\\ [:obj:`~typing.Union`\\ [%s]]' % args
            else:
                return ':py:obj:`~typing.Optional`\\ [%s]' % restify(cls.__args__[0], mode)
        else:
            args = ', '.join((restify(a, mode) for a in cls.__args__))
            return ':py:obj:`~typing.Union`\\ [%s]' % args
    elif inspect.isgenericalias(cls):
        if isinstance(cls.__origin__, typing._SpecialForm):
            text = restify(cls.__origin__, mode)
        elif getattr(cls, '_name', None):
            if cls.__module__ == 'typing':
                text = ':py:class:`~%s.%s`' % (cls.__module__, cls._name)
            else:
                text = ':py:class:`%s%s.%s`' % (modprefix, cls.__module__, cls._name)
        else:
            text = restify(cls.__origin__, mode)
        origin = getattr(cls, '__origin__', None)
        if not hasattr(cls, '__args__'):
            pass
        elif all((is_system_TypeVar(a) for a in cls.__args__)):
            pass
        elif cls.__module__ == 'typing' and cls._name == 'Callable':
            args = ', '.join((restify(a, mode) for a in cls.__args__[:-1]))
            text += '\\ [[%s], %s]' % (args, restify(cls.__args__[-1], mode))
        elif cls.__module__ == 'typing' and getattr(origin, '_name', None) == 'Literal':
            text += '\\ [%s]' % ', '.join((repr(a) for a in cls.__args__))
        elif cls.__args__:
            text += '\\ [%s]' % ', '.join((restify(a, mode) for a in cls.__args__))
        return text
    elif isinstance(cls, typing._SpecialForm):
        return ':py:obj:`~%s.%s`' % (cls.__module__, cls._name)
    elif sys.version_info >= (3, 11) and cls is typing.Any:
        return f':py:obj:`~{cls.__module__}.{cls.__name__}`'
    elif hasattr(cls, '__qualname__'):
        if cls.__module__ == 'typing':
            return ':py:class:`~%s.%s`' % (cls.__module__, cls.__qualname__)
        else:
            return ':py:class:`%s%s.%s`' % (modprefix, cls.__module__, cls.__qualname__)
    elif isinstance(cls, ForwardRef):
        return ':py:class:`%s`' % cls.__forward_arg__
    elif cls.__module__ == 'typing':
        return ':py:obj:`~%s.%s`' % (cls.__module__, cls.__name__)
    else:
        return ':py:obj:`%s%s.%s`' % (modprefix, cls.__module__, cls.__name__)
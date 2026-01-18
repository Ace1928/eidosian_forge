import sys
import typing
from struct import Struct
from types import TracebackType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union
from docutils import nodes
from docutils.parsers.rst.states import Inliner
from sphinx.deprecation import RemovedInSphinx60Warning, deprecated_alias
def _restify_py36(cls: Optional[Type], mode: str='fully-qualified-except-typing') -> str:
    if mode == 'smart':
        modprefix = '~'
    else:
        modprefix = ''
    module = getattr(cls, '__module__', None)
    if module == 'typing':
        if getattr(cls, '_name', None):
            qualname = cls._name
        elif getattr(cls, '__qualname__', None):
            qualname = cls.__qualname__
        elif getattr(cls, '__forward_arg__', None):
            qualname = cls.__forward_arg__
        elif getattr(cls, '__origin__', None):
            qualname = stringify(cls.__origin__)
        else:
            qualname = repr(cls).replace('typing.', '')
    elif hasattr(cls, '__qualname__'):
        qualname = '%s%s.%s' % (modprefix, module, cls.__qualname__)
    else:
        qualname = repr(cls)
    if isinstance(cls, typing.TupleMeta) and (not hasattr(cls, '__tuple_params__')):
        if module == 'typing':
            reftext = ':py:class:`~typing.%s`' % qualname
        else:
            reftext = ':py:class:`%s%s`' % (modprefix, qualname)
        params = cls.__args__
        if params:
            param_str = ', '.join((restify(p, mode) for p in params))
            return reftext + '\\ [%s]' % param_str
        else:
            return reftext
    elif isinstance(cls, typing.GenericMeta):
        if module == 'typing':
            reftext = ':py:class:`~typing.%s`' % qualname
        else:
            reftext = ':py:class:`%s%s`' % (modprefix, qualname)
        if cls.__args__ is None or len(cls.__args__) <= 2:
            params = cls.__args__
        elif cls.__origin__ == Generator:
            params = cls.__args__
        else:
            args = ', '.join((restify(arg, mode) for arg in cls.__args__[:-1]))
            result = restify(cls.__args__[-1], mode)
            return reftext + '\\ [[%s], %s]' % (args, result)
        if params:
            param_str = ', '.join((restify(p, mode) for p in params))
            return reftext + '\\ [%s]' % param_str
        else:
            return reftext
    elif hasattr(cls, '__origin__') and cls.__origin__ is typing.Union:
        params = cls.__args__
        if params is not None:
            if len(params) > 1 and params[-1] is NoneType:
                if len(params) > 2:
                    param_str = ', '.join((restify(p, mode) for p in params[:-1]))
                    return ':py:obj:`~typing.Optional`\\ [:py:obj:`~typing.Union`\\ [%s]]' % param_str
                else:
                    return ':py:obj:`~typing.Optional`\\ [%s]' % restify(params[0], mode)
            else:
                param_str = ', '.join((restify(p, mode) for p in params))
                return ':py:obj:`~typing.Union`\\ [%s]' % param_str
        else:
            return ':py:obj:`Union`'
    elif hasattr(cls, '__qualname__'):
        if cls.__module__ == 'typing':
            return ':py:class:`~%s.%s`' % (cls.__module__, cls.__qualname__)
        else:
            return ':py:class:`%s%s.%s`' % (modprefix, cls.__module__, cls.__qualname__)
    elif hasattr(cls, '_name'):
        if cls.__module__ == 'typing':
            return ':py:obj:`~%s.%s`' % (cls.__module__, cls._name)
        else:
            return ':py:obj:`%s%s.%s`' % (modprefix, cls.__module__, cls._name)
    elif hasattr(cls, '__name__'):
        if cls.__module__ == 'typing':
            return ':py:obj:`~%s.%s`' % (cls.__module__, cls.__name__)
        else:
            return ':py:obj:`%s%s.%s`' % (modprefix, cls.__module__, cls.__name__)
    elif cls.__module__ == 'typing':
        return ':py:obj:`~%s.%s`' % (cls.__module__, qualname)
    else:
        return ':py:obj:`%s%s.%s`' % (modprefix, cls.__module__, qualname)
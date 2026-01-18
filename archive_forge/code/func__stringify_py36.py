import sys
import typing
from struct import Struct
from types import TracebackType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union
from docutils import nodes
from docutils.parsers.rst.states import Inliner
from sphinx.deprecation import RemovedInSphinx60Warning, deprecated_alias
def _stringify_py36(annotation: Any, mode: str='fully-qualified-except-typing') -> str:
    """stringify() for py36."""
    module = getattr(annotation, '__module__', None)
    modprefix = ''
    if module == 'typing' and getattr(annotation, '__forward_arg__', None):
        qualname = annotation.__forward_arg__
    elif module == 'typing':
        if getattr(annotation, '_name', None):
            qualname = annotation._name
        elif getattr(annotation, '__qualname__', None):
            qualname = annotation.__qualname__
        elif getattr(annotation, '__origin__', None):
            qualname = stringify(annotation.__origin__)
        else:
            qualname = repr(annotation).replace('typing.', '')
        if mode == 'smart':
            modprefix = '~%s.' % module
        elif mode == 'fully-qualified':
            modprefix = '%s.' % module
    elif hasattr(annotation, '__qualname__'):
        if mode == 'smart':
            modprefix = '~%s.' % module
        else:
            modprefix = '%s.' % module
        qualname = annotation.__qualname__
    else:
        qualname = repr(annotation)
    if isinstance(annotation, typing.TupleMeta) and (not hasattr(annotation, '__tuple_params__')):
        params = annotation.__args__
        if params:
            param_str = ', '.join((stringify(p, mode) for p in params))
            return '%s%s[%s]' % (modprefix, qualname, param_str)
        else:
            return modprefix + qualname
    elif isinstance(annotation, typing.GenericMeta):
        params = None
        if annotation.__args__ is None or len(annotation.__args__) <= 2:
            params = annotation.__args__
        elif annotation.__origin__ == Generator:
            params = annotation.__args__
        else:
            args = ', '.join((stringify(arg, mode) for arg in annotation.__args__[:-1]))
            result = stringify(annotation.__args__[-1])
            return '%s%s[[%s], %s]' % (modprefix, qualname, args, result)
        if params is not None:
            param_str = ', '.join((stringify(p, mode) for p in params))
            return '%s%s[%s]' % (modprefix, qualname, param_str)
    elif hasattr(annotation, '__origin__') and annotation.__origin__ is typing.Union:
        params = annotation.__args__
        if params is not None:
            if len(params) > 1 and params[-1] is NoneType:
                if len(params) > 2:
                    param_str = ', '.join((stringify(p, mode) for p in params[:-1]))
                    return '%sOptional[%sUnion[%s]]' % (modprefix, modprefix, param_str)
                else:
                    return '%sOptional[%s]' % (modprefix, stringify(params[0], mode))
            else:
                param_str = ', '.join((stringify(p, mode) for p in params))
                return '%sUnion[%s]' % (modprefix, param_str)
    return modprefix + qualname
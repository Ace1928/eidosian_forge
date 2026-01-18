import sys
import typing
from struct import Struct
from types import TracebackType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union
from docutils import nodes
from docutils.parsers.rst.states import Inliner
from sphinx.deprecation import RemovedInSphinx60Warning, deprecated_alias
def _stringify_py37(annotation: Any, mode: str='fully-qualified-except-typing') -> str:
    """stringify() for py37+."""
    module = getattr(annotation, '__module__', None)
    modprefix = ''
    if module == 'typing' and getattr(annotation, '__forward_arg__', None):
        qualname = annotation.__forward_arg__
    elif module == 'typing':
        if getattr(annotation, '_name', None):
            qualname = annotation._name
        elif getattr(annotation, '__qualname__', None):
            qualname = annotation.__qualname__
        else:
            qualname = stringify(annotation.__origin__).replace('typing.', '')
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
    elif hasattr(annotation, '__origin__'):
        qualname = stringify(annotation.__origin__, mode)
    elif UnionType and isinstance(annotation, UnionType):
        qualname = 'types.Union'
    else:
        return repr(annotation)
    if getattr(annotation, '__args__', None):
        if not isinstance(annotation.__args__, (list, tuple)):
            pass
        elif qualname in ('Optional', 'Union'):
            if len(annotation.__args__) > 1 and annotation.__args__[-1] is NoneType:
                if len(annotation.__args__) > 2:
                    args = ', '.join((stringify(a, mode) for a in annotation.__args__[:-1]))
                    return '%sOptional[%sUnion[%s]]' % (modprefix, modprefix, args)
                else:
                    return '%sOptional[%s]' % (modprefix, stringify(annotation.__args__[0], mode))
            else:
                args = ', '.join((stringify(a, mode) for a in annotation.__args__))
                return '%sUnion[%s]' % (modprefix, args)
        elif qualname == 'types.Union':
            if len(annotation.__args__) > 1 and None in annotation.__args__:
                args = ' | '.join((stringify(a) for a in annotation.__args__ if a))
                return '%sOptional[%s]' % (modprefix, args)
            else:
                return ' | '.join((stringify(a) for a in annotation.__args__))
        elif qualname == 'Callable':
            args = ', '.join((stringify(a, mode) for a in annotation.__args__[:-1]))
            returns = stringify(annotation.__args__[-1], mode)
            return '%s%s[[%s], %s]' % (modprefix, qualname, args, returns)
        elif qualname == 'Literal':
            args = ', '.join((repr(a) for a in annotation.__args__))
            return '%s%s[%s]' % (modprefix, qualname, args)
        elif str(annotation).startswith('typing.Annotated'):
            return stringify(annotation.__args__[0], mode)
        elif all((is_system_TypeVar(a) for a in annotation.__args__)):
            return modprefix + qualname
        else:
            args = ', '.join((stringify(a, mode) for a in annotation.__args__))
            return '%s%s[%s]' % (modprefix, qualname, args)
    return modprefix + qualname
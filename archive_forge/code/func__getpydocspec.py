import inspect
import keyword
import pydoc
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Dict, List, ContextManager
from types import MemberDescriptorType, TracebackType
from ._typing_compat import Literal
from pygments.token import Token
from pygments.lexers import Python3Lexer
from .lazyre import LazyReCompile
def _getpydocspec(f: Callable) -> Optional[ArgSpec]:
    try:
        argspec = pydoc.getdoc(f)
    except NameError:
        return None
    s = _getpydocspec_re.search(argspec)
    if s is None:
        return None
    if not hasattr_safe(f, '__name__') or s.groups()[0] != f.__name__:
        return None
    args = []
    defaults = []
    varargs = varkwargs = None
    kwonly_args = []
    kwonly_defaults = {}
    for arg in s.group(2).split(','):
        arg = arg.strip()
        if arg.startswith('**'):
            varkwargs = arg[2:]
        elif arg.startswith('*'):
            varargs = arg[1:]
        elif arg == '...':
            varargs = ''
        else:
            arg, _, default = arg.partition('=')
            if varargs is not None:
                kwonly_args.append(arg)
                if default:
                    kwonly_defaults[arg] = _Repr(default)
            else:
                args.append(arg)
                if default:
                    defaults.append(_Repr(default))
    return ArgSpec(args, varargs, varkwargs, defaults, kwonly_args, kwonly_defaults, None)
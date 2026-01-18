from __future__ import annotations
import collections
from collections.abc import Iterable
import textwrap
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
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
import uuid
import warnings
from sqlalchemy.util import asbool as asbool  # noqa: F401
from sqlalchemy.util import immutabledict as immutabledict  # noqa: F401
from sqlalchemy.util import to_list as to_list  # noqa: F401
from sqlalchemy.util import unique_list as unique_list
from .compat import inspect_getfullargspec
@classmethod
def _create_method_proxy(cls, name: str, globals_: MutableMapping[str, Any], locals_: MutableMapping[str, Any]) -> Callable[..., Any]:
    fn = getattr(cls, name)

    def _name_error(name: str, from_: Exception) -> NoReturn:
        raise NameError("Can't invoke function '%s', as the proxy object has not yet been established for the Alembic '%s' class.  Try placing this code inside a callable." % (name, cls.__name__)) from from_
    globals_['_name_error'] = _name_error
    translations = getattr(fn, '_legacy_translations', [])
    if translations:
        spec = inspect_getfullargspec(fn)
        if spec[0] and spec[0][0] == 'self':
            spec[0].pop(0)
        outer_args = inner_args = '*args, **kw'
        translate_str = 'args, kw = _translate(%r, %r, %r, args, kw)' % (fn.__name__, tuple(spec), translations)

        def translate(fn_name: str, spec: Any, translations: Any, args: Any, kw: Any) -> Any:
            return_kw = {}
            return_args = []
            for oldname, newname in translations:
                if oldname in kw:
                    warnings.warn('Argument %r is now named %r for method %s().' % (oldname, newname, fn_name))
                    return_kw[newname] = kw.pop(oldname)
            return_kw.update(kw)
            args = list(args)
            if spec[3]:
                pos_only = spec[0][:-len(spec[3])]
            else:
                pos_only = spec[0]
            for arg in pos_only:
                if arg not in return_kw:
                    try:
                        return_args.append(args.pop(0))
                    except IndexError:
                        raise TypeError('missing required positional argument: %s' % arg)
            return_args.extend(args)
            return (return_args, return_kw)
        globals_['_translate'] = translate
    else:
        outer_args = '*args, **kw'
        inner_args = '*args, **kw'
        translate_str = ''
    func_text = textwrap.dedent("        def %(name)s(%(args)s):\n            %(doc)r\n            %(translate)s\n            try:\n                p = _proxy\n            except NameError as ne:\n                _name_error('%(name)s', ne)\n            return _proxy.%(name)s(%(apply_kw)s)\n            e\n        " % {'name': name, 'translate': translate_str, 'args': outer_args, 'apply_kw': inner_args, 'doc': fn.__doc__})
    lcl: MutableMapping[str, Any] = {}
    exec(func_text, cast('Dict[str, Any]', globals_), lcl)
    return cast('Callable[..., Any]', lcl[name])
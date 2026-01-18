from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from .registry import _ET
from .registry import _ListenerFnType
from .. import util
from ..util.compat import FullArgSpec
def _wrap_fn_for_legacy(dispatch_collection: _ClsLevelDispatch[_ET], fn: _ListenerFnType, argspec: FullArgSpec) -> _ListenerFnType:
    for since, argnames, conv in dispatch_collection.legacy_signatures:
        if argnames[-1] == '**kw':
            has_kw = True
            argnames = argnames[0:-1]
        else:
            has_kw = False
        if len(argnames) == len(argspec.args) and has_kw is bool(argspec.varkw):
            formatted_def = 'def %s(%s%s)' % (dispatch_collection.name, ', '.join(dispatch_collection.arg_names), ', **kw' if has_kw else '')
            warning_txt = 'The argument signature for the "%s.%s" event listener has changed as of version %s, and conversion for the old argument signature will be removed in a future release.  The new signature is "%s"' % (dispatch_collection.clsname, dispatch_collection.name, since, formatted_def)
            if conv is not None:
                assert not has_kw

                def wrap_leg(*args: Any, **kw: Any) -> Any:
                    util.warn_deprecated(warning_txt, version=since)
                    assert conv is not None
                    return fn(*conv(*args))
            else:

                def wrap_leg(*args: Any, **kw: Any) -> Any:
                    util.warn_deprecated(warning_txt, version=since)
                    argdict = dict(zip(dispatch_collection.arg_names, args))
                    args_from_dict = [argdict[name] for name in argnames]
                    if has_kw:
                        return fn(*args_from_dict, **kw)
                    else:
                        return fn(*args_from_dict)
            return wrap_leg
    else:
        return fn
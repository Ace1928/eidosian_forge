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
def _version_signature_changes(parent_dispatch_cls: Type[_HasEventsDispatch[_ET]], dispatch_collection: _ClsLevelDispatch[_ET]) -> str:
    since, args, conv = dispatch_collection.legacy_signatures[0]
    return '\n.. versionchanged:: %(since)s\n    The :meth:`.%(clsname)s.%(event_name)s` event now accepts the \n    arguments %(named_event_arguments)s%(has_kw_arguments)s.\n    Support for listener functions which accept the previous \n    argument signature(s) listed above as "deprecated" will be \n    removed in a future release.' % {'since': since, 'clsname': parent_dispatch_cls.__name__, 'event_name': dispatch_collection.name, 'named_event_arguments': ', '.join((':paramref:`.%(clsname)s.%(event_name)s.%(param_name)s`' % {'clsname': parent_dispatch_cls.__name__, 'event_name': dispatch_collection.name, 'param_name': param_name} for param_name in dispatch_collection.arg_names)), 'has_kw_arguments': ', **kw' if dispatch_collection.has_kw else ''}
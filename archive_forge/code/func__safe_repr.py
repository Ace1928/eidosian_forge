import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def _safe_repr(self, object, context, maxlevels, level):
    typ = type(object)
    if typ in _builtin_scalars:
        return (repr(object), True, False)
    r = getattr(typ, '__repr__', None)
    if issubclass(typ, int) and r is int.__repr__:
        if self._underscore_numbers:
            return (f'{object:_d}', True, False)
        else:
            return (repr(object), True, False)
    if issubclass(typ, dict) and r is dict.__repr__:
        if not object:
            return ('{}', True, False)
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return ('{...}', False, objid in context)
        if objid in context:
            return (_recursion(object), False, True)
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        if self._sort_dicts:
            items = sorted(object.items(), key=_safe_tuple)
        else:
            items = object.items()
        for k, v in items:
            krepr, kreadable, krecur = self.format(k, context, maxlevels, level)
            vrepr, vreadable, vrecur = self.format(v, context, maxlevels, level)
            append('%s: %s' % (krepr, vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return ('{%s}' % ', '.join(components), readable, recursive)
    if issubclass(typ, list) and r is list.__repr__ or (issubclass(typ, tuple) and r is tuple.__repr__):
        if issubclass(typ, list):
            if not object:
                return ('[]', True, False)
            format = '[%s]'
        elif len(object) == 1:
            format = '(%s,)'
        else:
            if not object:
                return ('()', True, False)
            format = '(%s)'
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return (format % '...', False, objid in context)
        if objid in context:
            return (_recursion(object), False, True)
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        for o in object:
            orepr, oreadable, orecur = self.format(o, context, maxlevels, level)
            append(orepr)
            if not oreadable:
                readable = False
            if orecur:
                recursive = True
        del context[objid]
        return (format % ', '.join(components), readable, recursive)
    rep = repr(object)
    return (rep, rep and (not rep.startswith('<')), False)
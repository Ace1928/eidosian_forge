from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _get_struct_union_enum_type(self, kind, type, name=None, nested=False):
    try:
        return self._structnode2type[type]
    except KeyError:
        pass
    force_name = name
    name = type.name
    if name is None:
        if force_name is not None:
            explicit_name = '$%s' % force_name
        else:
            self._anonymous_counter += 1
            explicit_name = '$%d' % self._anonymous_counter
        tp = None
    else:
        explicit_name = name
        key = '%s %s' % (kind, name)
        tp, _ = self._declarations.get(key, (None, None))
    if tp is None:
        if kind == 'struct':
            tp = model.StructType(explicit_name, None, None, None)
        elif kind == 'union':
            tp = model.UnionType(explicit_name, None, None, None)
        elif kind == 'enum':
            if explicit_name == '__dotdotdot__':
                raise CDefError('Enums cannot be declared with ...')
            tp = self._build_enum_type(explicit_name, type.values)
        else:
            raise AssertionError('kind = %r' % (kind,))
        if name is not None:
            self._declare(key, tp)
    elif kind == 'enum' and type.values is not None:
        raise NotImplementedError("enum %s: the '{}' declaration should appear on the first time the enum is mentioned, not later" % explicit_name)
    if not tp.forcename:
        tp.force_the_name(force_name)
    if tp.forcename and '$' in tp.name:
        self._declare('anonymous %s' % tp.forcename, tp)
    self._structnode2type[type] = tp
    if kind == 'enum':
        return tp
    if type.decls is None:
        return tp
    if tp.fldnames is not None:
        raise CDefError('duplicate declaration of struct %s' % name)
    fldnames = []
    fldtypes = []
    fldbitsize = []
    fldquals = []
    for decl in type.decls:
        if isinstance(decl.type, pycparser.c_ast.IdentifierType) and ''.join(decl.type.names) == '__dotdotdot__':
            self._make_partial(tp, nested)
            continue
        if decl.bitsize is None:
            bitsize = -1
        else:
            bitsize = self._parse_constant(decl.bitsize)
        self._partial_length = False
        type, fqual = self._get_type_and_quals(decl.type, partial_length_ok=True)
        if self._partial_length:
            self._make_partial(tp, nested)
        if isinstance(type, model.StructType) and type.partial:
            self._make_partial(tp, nested)
        fldnames.append(decl.name or '')
        fldtypes.append(type)
        fldbitsize.append(bitsize)
        fldquals.append(fqual)
    tp.fldnames = tuple(fldnames)
    tp.fldtypes = tuple(fldtypes)
    tp.fldbitsize = tuple(fldbitsize)
    tp.fldquals = tuple(fldquals)
    if fldbitsize != [-1] * len(fldbitsize):
        if isinstance(tp, model.StructType) and tp.partial:
            raise NotImplementedError("%s: using both bitfields and '...;'" % (tp,))
    tp.packed = self._options.get('packed')
    if tp.completed:
        tp.completed = 0
        self._recomplete.append(tp)
    return tp
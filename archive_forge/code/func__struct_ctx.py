import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _struct_ctx(self, tp, cname, approxname, named_ptr=None):
    type_index = self._typesdict[tp]
    reason_for_not_expanding = None
    flags = []
    if isinstance(tp, model.UnionType):
        flags.append('_CFFI_F_UNION')
    if tp.fldtypes is None:
        flags.append('_CFFI_F_OPAQUE')
        reason_for_not_expanding = 'opaque'
    if tp not in self.ffi._parser._included_declarations and (named_ptr is None or named_ptr not in self.ffi._parser._included_declarations):
        if tp.fldtypes is None:
            pass
        elif tp.partial or any(tp.anonymous_struct_fields()):
            pass
        else:
            flags.append('_CFFI_F_CHECK_FIELDS')
        if tp.packed:
            if tp.packed > 1:
                raise NotImplementedError('%r is declared with \'pack=%r\'; only 0 or 1 are supported in API mode (try to use "...;", which does not require a \'pack\' declaration)' % (tp, tp.packed))
            flags.append('_CFFI_F_PACKED')
    else:
        flags.append('_CFFI_F_EXTERNAL')
        reason_for_not_expanding = 'external'
    flags = '|'.join(flags) or '0'
    c_fields = []
    if reason_for_not_expanding is None:
        enumfields = list(self._enum_fields(tp))
        for fldname, fldtype, fbitsize, fqual in enumfields:
            fldtype = self._field_type(tp, fldname, fldtype)
            self._check_not_opaque(fldtype, "field '%s.%s'" % (tp.name, fldname))
            op = OP_NOOP
            if fbitsize >= 0:
                op = OP_BITFIELD
                size = '%d /* bits */' % fbitsize
            elif cname is None or (isinstance(fldtype, model.ArrayType) and fldtype.length is None):
                size = '(size_t)-1'
            else:
                size = 'sizeof(((%s)0)->%s)' % (tp.get_c_name('*') if named_ptr is None else named_ptr.name, fldname)
            if cname is None or fbitsize >= 0:
                offset = '(size_t)-1'
            elif named_ptr is not None:
                offset = '((char *)&((%s)0)->%s) - (char *)0' % (named_ptr.name, fldname)
            else:
                offset = 'offsetof(%s, %s)' % (tp.get_c_name(''), fldname)
            c_fields.append(FieldExpr(fldname, offset, size, fbitsize, CffiOp(op, self._typesdict[fldtype])))
        first_field_index = len(self._lsts['field'])
        self._lsts['field'].extend(c_fields)
        if cname is None:
            size = '(size_t)-2'
            align = -2
            comment = 'unnamed'
        else:
            if named_ptr is not None:
                size = 'sizeof(*(%s)0)' % (named_ptr.name,)
                align = '-1 /* unknown alignment */'
            else:
                size = 'sizeof(%s)' % (cname,)
                align = 'offsetof(struct _cffi_align_%s, y)' % (approxname,)
            comment = None
    else:
        size = '(size_t)-1'
        align = -1
        first_field_index = -1
        comment = reason_for_not_expanding
    self._lsts['struct_union'].append(StructUnionExpr(tp.name, type_index, flags, size, align, comment, first_field_index, c_fields))
    self._seen_struct_unions.add(tp)
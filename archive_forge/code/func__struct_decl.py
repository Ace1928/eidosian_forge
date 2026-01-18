import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _struct_decl(self, tp, cname, approxname):
    if tp.fldtypes is None:
        return
    prnt = self._prnt
    checkfuncname = '_cffi_checkfld_%s' % (approxname,)
    prnt('_CFFI_UNUSED_FN')
    prnt('static void %s(%s *p)' % (checkfuncname, cname))
    prnt('{')
    prnt('  /* only to generate compile-time warnings or errors */')
    prnt('  (void)p;')
    for fname, ftype, fbitsize, fqual in self._enum_fields(tp):
        try:
            if ftype.is_integer_type() or fbitsize >= 0:
                if fname != '':
                    prnt("  (void)((p->%s) | 0);  /* check that '%s.%s' is an integer */" % (fname, cname, fname))
                continue
            while isinstance(ftype, model.ArrayType) and (ftype.length is None or ftype.length == '...'):
                ftype = ftype.item
                fname = fname + '[0]'
            prnt('  { %s = &p->%s; (void)tmp; }' % (ftype.get_c_name('*tmp', 'field %r' % fname, quals=fqual), fname))
        except VerificationError as e:
            prnt('  /* %s */' % str(e))
    prnt('}')
    prnt('struct _cffi_align_%s { char x; %s y; };' % (approxname, cname))
    prnt()
from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
class PyObjectType(PyrexType):
    name = 'object'
    is_pyobject = 1
    default_value = '0'
    declaration_value = '0'
    buffer_defaults = None
    is_external = False
    is_subclassed = False
    is_gc_simple = False
    builtin_trashcan = False
    needs_refcounting = True

    def __str__(self):
        return 'Python object'

    def __repr__(self):
        return '<PyObjectType>'

    def can_coerce_to_pyobject(self, env):
        return True

    def can_coerce_from_pyobject(self, env):
        return True

    def can_be_optional(self):
        """Returns True if type can be used with typing.Optional[]."""
        return True

    def default_coerced_ctype(self):
        """The default C type that this Python type coerces to, or None."""
        return None

    def assignable_from(self, src_type):
        return not src_type.is_ptr or src_type.is_string or src_type.is_pyunicode_ptr

    def is_simple_buffer_dtype(self):
        return True

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        if pyrex or for_display:
            base_code = 'object'
        else:
            base_code = public_decl('PyObject', dll_linkage)
            entity_code = '*%s' % entity_code
        return self.base_declaration_code(base_code, entity_code)

    def as_pyobject(self, cname):
        if not self.is_complete() or self.is_extension_type:
            return '(PyObject *)' + cname
        else:
            return cname

    def py_type_name(self):
        return 'object'

    def __lt__(self, other):
        """
        Make sure we sort highest, as instance checking on py_type_name
        ('object') is always true
        """
        return False

    def global_init_code(self, entry, code):
        code.put_init_var_to_py_none(entry, nanny=False)

    def check_for_null_code(self, cname):
        return cname

    def generate_incref(self, code, cname, nanny):
        if nanny:
            code.funcstate.needs_refnanny = True
            code.putln('__Pyx_INCREF(%s);' % self.as_pyobject(cname))
        else:
            code.putln('Py_INCREF(%s);' % self.as_pyobject(cname))

    def generate_xincref(self, code, cname, nanny):
        if nanny:
            code.funcstate.needs_refnanny = True
            code.putln('__Pyx_XINCREF(%s);' % self.as_pyobject(cname))
        else:
            code.putln('Py_XINCREF(%s);' % self.as_pyobject(cname))

    def generate_decref(self, code, cname, nanny, have_gil):
        assert have_gil
        self._generate_decref(code, cname, nanny, null_check=False, clear=False)

    def generate_xdecref(self, code, cname, nanny, have_gil):
        self._generate_decref(code, cname, nanny, null_check=True, clear=False)

    def generate_decref_clear(self, code, cname, clear_before_decref, nanny, have_gil):
        self._generate_decref(code, cname, nanny, null_check=False, clear=True, clear_before_decref=clear_before_decref)

    def generate_xdecref_clear(self, code, cname, clear_before_decref=False, nanny=True, have_gil=None):
        self._generate_decref(code, cname, nanny, null_check=True, clear=True, clear_before_decref=clear_before_decref)

    def generate_gotref(self, code, cname):
        code.funcstate.needs_refnanny = True
        code.putln('__Pyx_GOTREF(%s);' % self.as_pyobject(cname))

    def generate_xgotref(self, code, cname):
        code.funcstate.needs_refnanny = True
        code.putln('__Pyx_XGOTREF(%s);' % self.as_pyobject(cname))

    def generate_giveref(self, code, cname):
        code.funcstate.needs_refnanny = True
        code.putln('__Pyx_GIVEREF(%s);' % self.as_pyobject(cname))

    def generate_xgiveref(self, code, cname):
        code.funcstate.needs_refnanny = True
        code.putln('__Pyx_XGIVEREF(%s);' % self.as_pyobject(cname))

    def generate_decref_set(self, code, cname, rhs_cname):
        code.funcstate.needs_refnanny = True
        code.putln('__Pyx_DECREF_SET(%s, %s);' % (cname, rhs_cname))

    def generate_xdecref_set(self, code, cname, rhs_cname):
        code.funcstate.needs_refnanny = True
        code.putln('__Pyx_XDECREF_SET(%s, %s);' % (cname, rhs_cname))

    def _generate_decref(self, code, cname, nanny, null_check=False, clear=False, clear_before_decref=False):
        prefix = '__Pyx' if nanny else 'Py'
        X = 'X' if null_check else ''
        if nanny:
            code.funcstate.needs_refnanny = True
        if clear:
            if clear_before_decref:
                if not nanny:
                    X = ''
                code.putln('%s_%sCLEAR(%s);' % (prefix, X, cname))
            else:
                code.putln('%s_%sDECREF(%s); %s = 0;' % (prefix, X, self.as_pyobject(cname), cname))
        else:
            code.putln('%s_%sDECREF(%s);' % (prefix, X, self.as_pyobject(cname)))

    def nullcheck_string(self, cname):
        return cname
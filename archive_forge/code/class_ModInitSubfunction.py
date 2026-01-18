from __future__ import absolute_import
import cython
from collections import defaultdict
import json
import operator
import os
import re
import sys
from .PyrexTypes import CPtrType
from . import Future
from . import Annotate
from . import Code
from . import Naming
from . import Nodes
from . import Options
from . import TypeSlots
from . import PyrexTypes
from . import Pythran
from .Errors import error, warning, CompileError
from .PyrexTypes import py_object_type
from ..Utils import open_new_file, replace_suffix, decode_filename, build_hex_version, is_cython_generated_file
from .Code import UtilityCode, IncludeCode, TempitaUtilityCode
from .StringEncoding import EncodedString, encoded_string_or_bytes_literal
from .Pythran import has_np_pythran
class ModInitSubfunction(object):

    def __init__(self, code_type):
        cname = '_'.join(code_type.lower().split())
        assert re.match('^[a-z0-9_]+$', cname)
        self.cfunc_name = '__Pyx_modinit_%s' % cname
        self.description = code_type
        self.tempdecl_code = None
        self.call_code = None

    def __enter__(self):
        self.call_code = orig_code.insertion_point()
        code = function_code
        code.enter_cfunc_scope(scope)
        prototypes.putln('static CYTHON_SMALL_CODE int %s(void); /*proto*/' % self.cfunc_name)
        code.putln('static int %s(void) {' % self.cfunc_name)
        code.put_declare_refcount_context()
        self.tempdecl_code = code.insertion_point()
        code.put_setup_refcount_context(EncodedString(self.cfunc_name))
        code.putln('/*--- %s ---*/' % self.description)
        return code

    def __exit__(self, *args):
        code = function_code
        code.put_finish_refcount_context()
        code.putln('return 0;')
        self.tempdecl_code.put_temp_declarations(code.funcstate)
        self.tempdecl_code = None
        needs_error_handling = code.label_used(code.error_label)
        if needs_error_handling:
            code.put_label(code.error_label)
            for cname, type in code.funcstate.all_managed_temps():
                code.put_xdecref(cname, type)
            code.put_finish_refcount_context()
            code.putln('return -1;')
        code.putln('}')
        code.exit_cfunc_scope()
        code.putln('')
        if needs_error_handling:
            self.call_code.putln(self.call_code.error_goto_if_neg('%s()' % self.cfunc_name, pos))
        else:
            self.call_code.putln('(void)%s();' % self.cfunc_name)
        self.call_code = None
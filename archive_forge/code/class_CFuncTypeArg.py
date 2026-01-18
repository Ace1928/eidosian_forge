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
class CFuncTypeArg(BaseType):
    not_none = False
    or_none = False
    accept_none = True
    accept_builtin_subtypes = False
    annotation = None
    subtypes = ['type']

    def __init__(self, name, type, pos, cname=None, annotation=None):
        self.name = name
        if cname is not None:
            self.cname = cname
        else:
            self.cname = Naming.var_prefix + name
        if annotation is not None:
            self.annotation = annotation
        self.type = type
        self.pos = pos
        self.needs_type_test = False

    def __repr__(self):
        return '%s:%s' % (self.name, repr(self.type))

    def declaration_code(self, for_display=0):
        return self.type.declaration_code(self.cname, for_display)

    def specialize(self, values):
        return CFuncTypeArg(self.name, self.type.specialize(values), self.pos, self.cname)

    def is_forwarding_reference(self):
        if self.type.is_rvalue_reference:
            if isinstance(self.type.ref_base_type, TemplatePlaceholderType) and (not self.type.ref_base_type.is_cv_qualified):
                return True
        return False
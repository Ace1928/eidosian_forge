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
class CIntType(CIntLike, CNumericType):
    is_int = 1
    typedef_flag = 0
    exception_value = -1

    def get_to_py_type_conversion(self):
        if self.rank < list(rank_to_type_name).index('int'):
            return 'PyInt_FromLong'
        else:
            Prefix = 'Int'
            SignWord = ''
            TypeName = 'Long'
            if not self.signed:
                Prefix = 'Long'
                SignWord = 'Unsigned'
            if self.rank >= list(rank_to_type_name).index('PY_LONG_LONG'):
                Prefix = 'Long'
                TypeName = 'LongLong'
            return 'Py%s_From%s%s' % (Prefix, SignWord, TypeName)

    def assignable_from_resolved_type(self, src_type):
        return src_type.is_int or src_type.is_enum or src_type is error_type

    def invalid_value(self):
        if rank_to_type_name[int(self.rank)] == 'char':
            return "'?'"
        else:
            return '0xbad0bad0'

    def overflow_check_binop(self, binop, env, const_rhs=False):
        env.use_utility_code(UtilityCode.load('Common', 'Overflow.c'))
        type = self.empty_declaration_code()
        name = self.specialization_name()
        if binop == 'lshift':
            env.use_utility_code(TempitaUtilityCode.load_cached('LeftShift', 'Overflow.c', context={'TYPE': type, 'NAME': name, 'SIGNED': self.signed}))
        else:
            if const_rhs:
                binop += '_const'
            if type in ('int', 'long', 'long long'):
                env.use_utility_code(TempitaUtilityCode.load_cached('BaseCaseSigned', 'Overflow.c', context={'INT': type, 'NAME': name}))
            elif type in ('unsigned int', 'unsigned long', 'unsigned long long'):
                env.use_utility_code(TempitaUtilityCode.load_cached('BaseCaseUnsigned', 'Overflow.c', context={'UINT': type, 'NAME': name}))
            elif self.rank <= 1:
                return '__Pyx_%s_%s_no_overflow' % (binop, name)
            else:
                _load_overflow_base(env)
                env.use_utility_code(TempitaUtilityCode.load_cached('SizeCheck', 'Overflow.c', context={'TYPE': type, 'NAME': name}))
                env.use_utility_code(TempitaUtilityCode.load_cached('Binop', 'Overflow.c', context={'TYPE': type, 'NAME': name, 'BINOP': binop}))
        return '__Pyx_%s_%s_checking_overflow' % (binop, name)
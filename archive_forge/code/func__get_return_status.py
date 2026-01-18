from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def _get_return_status(self, builder, code, excinfoptr):
    """
        Given a return *code* and *excinfoptr*, get a Status instance.
        """
    norm = builder.icmp_signed('==', code, RETCODE_OK)
    none = builder.icmp_signed('==', code, RETCODE_NONE)
    exc = builder.icmp_signed('==', code, RETCODE_EXC)
    is_stop_iteration = builder.icmp_signed('==', code, RETCODE_STOPIT)
    ok = builder.or_(norm, none)
    err = builder.not_(ok)
    is_user_exc = builder.icmp_signed('>=', code, RETCODE_USEREXC)
    excinfoptr = builder.select(is_user_exc, excinfoptr, ir.Constant(excinfo_ptr_t, ir.Undefined))
    status = Status(code=code, is_ok=ok, is_error=err, is_python_exc=exc, is_none=none, is_user_exc=is_user_exc, is_stop_iteration=is_stop_iteration, excinfoptr=excinfoptr)
    return status
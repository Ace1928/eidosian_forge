import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _make_c_or_py_source(ffi, module_name, preamble, target_file, verbose):
    if verbose:
        print('generating %s' % (target_file,))
    recompiler = Recompiler(ffi, module_name, target_is_python=preamble is None)
    recompiler.collect_type_table()
    recompiler.collect_step_tables()
    f = NativeIO()
    recompiler.write_source_to_f(f, preamble)
    output = f.getvalue()
    try:
        with open(target_file, 'r') as f1:
            if f1.read(len(output) + 1) != output:
                raise IOError
        if verbose:
            print('(already up-to-date)')
        return False
    except IOError:
        tmp_file = '%s.~%d' % (target_file, os.getpid())
        with open(tmp_file, 'w') as f1:
            f1.write(output)
        try:
            os.rename(tmp_file, target_file)
        except OSError:
            os.unlink(target_file)
            os.rename(tmp_file, target_file)
        return True
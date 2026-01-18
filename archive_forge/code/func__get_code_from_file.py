importers when locating support scripts as well as when importing modules.
import sys
import importlib.machinery # importlib first so we can test #15386 via -m
import importlib.util
import io
import os
def _get_code_from_file(run_name, fname):
    from pkgutil import read_code
    decoded_path = os.path.abspath(os.fsdecode(fname))
    with io.open_code(decoded_path) as f:
        code = read_code(f)
    if code is None:
        with io.open_code(decoded_path) as f:
            code = compile(f.read(), fname, 'exec')
    return (code, fname)
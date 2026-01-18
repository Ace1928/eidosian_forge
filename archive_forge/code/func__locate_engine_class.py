import sys, os, binascii, shutil, io
from . import __version_verifier_modules__
from . import ffiplatform
from .error import VerificationError
def _locate_engine_class(ffi, force_generic_engine):
    if _FORCE_GENERIC_ENGINE:
        force_generic_engine = True
    if not force_generic_engine:
        if '__pypy__' in sys.builtin_module_names:
            force_generic_engine = True
        else:
            try:
                import _cffi_backend
            except ImportError:
                _cffi_backend = '?'
            if ffi._backend is not _cffi_backend:
                force_generic_engine = True
    if force_generic_engine:
        from . import vengine_gen
        return vengine_gen.VGenericEngine
    else:
        from . import vengine_cpy
        return vengine_cpy.VCPythonEngine
import sys
import marshal
import contextlib
import dis
from . import _imp
from ._imp import find_module, PY_COMPILED, PY_FROZEN, PY_SOURCE
from .extern.packaging.version import Version
def _update_globals():
    """
    Patch the globals to remove the objects not available on some platforms.

    XXX it'd be better to test assertions about bytecode instead.
    """
    if not sys.platform.startswith('java') and sys.platform != 'cli':
        return
    incompatible = ('extract_constant', 'get_module_constant')
    for name in incompatible:
        del globals()[name]
        __all__.remove(name)
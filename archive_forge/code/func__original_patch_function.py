from __future__ import annotations
import sys
import eventlet
def _original_patch_function(func, *module_names):
    """Kind of the contrapositive of patch_function: decorates a
    function such that when it's called, sys.modules is populated only
    with the unpatched versions of the specified modules.  Unlike
    patch_function, only the names of the modules need be supplied,
    and there are no defaults.  This is a gross hack; tell your kids not
    to import inside function bodies!"""

    def patched(*args, **kw):
        saver = SysModulesSaver(module_names)
        for name in module_names:
            sys.modules[name] = original(name)
        try:
            return func(*args, **kw)
        finally:
            saver.restore()
    return patched
from __future__ import annotations
import sys
import eventlet
class SysModulesSaver:
    """Class that captures some subset of the current state of
    sys.modules.  Pass in an iterator of module names to the
    constructor."""

    def __init__(self, module_names=()):
        self._saved = {}
        imp.acquire_lock()
        self.save(*module_names)

    def save(self, *module_names):
        """Saves the named modules to the object."""
        for modname in module_names:
            self._saved[modname] = sys.modules.get(modname, None)

    def restore(self):
        """Restores the modules that the saver knows about into
        sys.modules.
        """
        try:
            for modname, mod in self._saved.items():
                if mod is not None:
                    sys.modules[modname] = mod
                else:
                    try:
                        del sys.modules[modname]
                    except KeyError:
                        pass
        finally:
            imp.release_lock()
import builtins as builtin_mod
from traitlets.config.configurable import Configurable
from traitlets import Instance
def add_builtin(self, key, value):
    """Add a builtin and save the original."""
    bdict = builtin_mod.__dict__
    orig = bdict.get(key, BuiltinUndefined)
    if value is HideBuiltin:
        if orig is not BuiltinUndefined:
            self._orig_builtins[key] = orig
            del bdict[key]
    else:
        self._orig_builtins[key] = orig
        bdict[key] = value
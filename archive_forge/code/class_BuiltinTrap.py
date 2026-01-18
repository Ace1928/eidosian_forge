import builtins as builtin_mod
from traitlets.config.configurable import Configurable
from traitlets import Instance
class BuiltinTrap(Configurable):
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)

    def __init__(self, shell=None):
        super(BuiltinTrap, self).__init__(shell=shell, config=None)
        self._orig_builtins = {}
        self._nested_level = 0
        self.shell = shell
        self.auto_builtins = {'exit': HideBuiltin, 'quit': HideBuiltin, 'get_ipython': self.shell.get_ipython}

    def __enter__(self):
        if self._nested_level == 0:
            self.activate()
        self._nested_level += 1
        return self

    def __exit__(self, type, value, traceback):
        if self._nested_level == 1:
            self.deactivate()
        self._nested_level -= 1
        return False

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

    def remove_builtin(self, key, orig):
        """Remove an added builtin and re-set the original."""
        if orig is BuiltinUndefined:
            del builtin_mod.__dict__[key]
        else:
            builtin_mod.__dict__[key] = orig

    def activate(self):
        """Store ipython references in the __builtin__ namespace."""
        add_builtin = self.add_builtin
        for name, func in self.auto_builtins.items():
            add_builtin(name, func)

    def deactivate(self):
        """Remove any builtins which might have been added by add_builtins, or
        restore overwritten ones to their previous values."""
        remove_builtin = self.remove_builtin
        for key, val in self._orig_builtins.items():
            remove_builtin(key, val)
        self._orig_builtins.clear()
        self._builtins_added = False
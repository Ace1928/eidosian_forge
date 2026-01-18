import builtins as builtin_mod
from traitlets.config.configurable import Configurable
from traitlets import Instance
Remove any builtins which might have been added by add_builtins, or
        restore overwritten ones to their previous values.
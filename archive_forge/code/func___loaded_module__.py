import importlib
import sys
@property
def __loaded_module__(self):
    """Load the module, or retrieve it if already loaded."""
    super_getattr = super().__getattribute__
    name = super_getattr('__module_name__')
    try:
        return sys.modules[name]
    except KeyError:
        importlib.import_module(name)
        return sys.modules[name]
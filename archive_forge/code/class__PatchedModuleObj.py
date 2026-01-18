from importlib import import_module
from .logging import get_logger
class _PatchedModuleObj:
    """Set all the modules components as attributes of the _PatchedModuleObj object."""

    def __init__(self, module, attrs=None):
        attrs = attrs or []
        if module is not None:
            for key in module.__dict__:
                if key in attrs or not key.startswith('__'):
                    setattr(self, key, getattr(module, key))
        self._original_module = module._original_module if isinstance(module, _PatchedModuleObj) else module
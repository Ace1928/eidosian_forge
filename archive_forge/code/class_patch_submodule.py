from importlib import import_module
from .logging import get_logger
class patch_submodule:
    """
    Patch a submodule attribute of an object, by keeping all other submodules intact at all levels.

    Example::

        >>> import importlib
        >>> from datasets.load import dataset_module_factory
        >>> from datasets.streaming import patch_submodule, xjoin
        >>>
        >>> dataset_module = dataset_module_factory("snli")
        >>> snli_module = importlib.import_module(dataset_module.module_path)
        >>> patcher = patch_submodule(snli_module, "os.path.join", xjoin)
        >>> patcher.start()
        >>> assert snli_module.os.path.join is xjoin
    """
    _active_patches = []

    def __init__(self, obj, target: str, new, attrs=None):
        self.obj = obj
        self.target = target
        self.new = new
        self.key = target.split('.')[0]
        self.original = {}
        self.attrs = attrs or []

    def __enter__(self):
        *submodules, target_attr = self.target.split('.')
        for i in range(len(submodules)):
            try:
                submodule = import_module('.'.join(submodules[:i + 1]))
            except ModuleNotFoundError:
                continue
            for attr in self.obj.__dir__():
                obj_attr = getattr(self.obj, attr)
                if obj_attr is submodule or (isinstance(obj_attr, _PatchedModuleObj) and obj_attr._original_module is submodule):
                    self.original[attr] = obj_attr
                    setattr(self.obj, attr, _PatchedModuleObj(obj_attr, attrs=self.attrs))
                    patched = getattr(self.obj, attr)
                    for key in submodules[i + 1:]:
                        setattr(patched, key, _PatchedModuleObj(getattr(patched, key, None), attrs=self.attrs))
                        patched = getattr(patched, key)
                    setattr(patched, target_attr, self.new)
        if submodules:
            try:
                attr_value = getattr(import_module('.'.join(submodules)), target_attr)
            except (AttributeError, ModuleNotFoundError):
                return
            for attr in self.obj.__dir__():
                if getattr(self.obj, attr) is attr_value:
                    self.original[attr] = getattr(self.obj, attr)
                    setattr(self.obj, attr, self.new)
        elif target_attr in globals()['__builtins__']:
            self.original[target_attr] = globals()['__builtins__'][target_attr]
            setattr(self.obj, target_attr, self.new)
        else:
            raise RuntimeError(f'Tried to patch attribute {target_attr} instead of a submodule.')

    def __exit__(self, *exc_info):
        for attr in list(self.original):
            setattr(self.obj, attr, self.original.pop(attr))

    def start(self):
        """Activate a patch."""
        self.__enter__()
        self._active_patches.append(self)

    def stop(self):
        """Stop an active patch."""
        try:
            self._active_patches.remove(self)
        except ValueError:
            return None
        return self.__exit__()
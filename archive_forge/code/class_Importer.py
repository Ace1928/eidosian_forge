import importlib
from abc import ABC, abstractmethod
from pickle import (  # type: ignore[attr-defined]  # type: ignore[attr-defined]
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
from ._mangling import demangle, get_mangle_prefix, is_mangled
class Importer(ABC):
    """Represents an environment to import modules from.

    By default, you can figure out what module an object belongs by checking
    __module__ and importing the result using __import__ or importlib.import_module.

    torch.package introduces module importers other than the default one.
    Each PackageImporter introduces a new namespace. Potentially a single
    name (e.g. 'foo.bar') is present in multiple namespaces.

    It supports two main operations:
        import_module: module_name -> module object
        get_name: object -> (parent module name, name of obj within module)

    The guarantee is that following round-trip will succeed or throw an ObjNotFoundError/ObjMisMatchError.
        module_name, obj_name = env.get_name(obj)
        module = env.import_module(module_name)
        obj2 = getattr(module, obj_name)
        assert obj1 is obj2
    """
    modules: Dict[str, ModuleType]

    @abstractmethod
    def import_module(self, module_name: str) -> ModuleType:
        """Import `module_name` from this environment.

        The contract is the same as for importlib.import_module.
        """
        pass

    def get_name(self, obj: Any, name: Optional[str]=None) -> Tuple[str, str]:
        """Given an object, return a name that can be used to retrieve the
        object from this environment.

        Args:
            obj: An object to get the module-environment-relative name for.
            name: If set, use this name instead of looking up __name__ or __qualname__ on `obj`.
                This is only here to match how Pickler handles __reduce__ functions that return a string,
                don't use otherwise.
        Returns:
            A tuple (parent_module_name, attr_name) that can be used to retrieve `obj` from this environment.
            Use it like:
                mod = importer.import_module(parent_module_name)
                obj = getattr(mod, attr_name)

        Raises:
            ObjNotFoundError: we couldn't retrieve `obj by name.
            ObjMisMatchError: we found a different object with the same name as `obj`.
        """
        if name is None and obj and (_Pickler.dispatch.get(type(obj)) is None):
            reduce = getattr(obj, '__reduce__', None)
            if reduce is not None:
                try:
                    rv = reduce()
                    if isinstance(rv, str):
                        name = rv
                except Exception:
                    pass
        if name is None:
            name = getattr(obj, '__qualname__', None)
        if name is None:
            name = obj.__name__
        orig_module_name = self.whichmodule(obj, name)
        module_name = demangle(orig_module_name)
        try:
            module = self.import_module(module_name)
            obj2, _ = _getattribute(module, name)
        except (ImportError, KeyError, AttributeError):
            raise ObjNotFoundError(f'{obj} was not found as {module_name}.{name}') from None
        if obj is obj2:
            return (module_name, name)

        def get_obj_info(obj):
            assert name is not None
            module_name = self.whichmodule(obj, name)
            is_mangled_ = is_mangled(module_name)
            location = get_mangle_prefix(module_name) if is_mangled_ else 'the current Python environment'
            importer_name = f'the importer for {get_mangle_prefix(module_name)}' if is_mangled_ else "'sys_importer'"
            return (module_name, location, importer_name)
        obj_module_name, obj_location, obj_importer_name = get_obj_info(obj)
        obj2_module_name, obj2_location, obj2_importer_name = get_obj_info(obj2)
        msg = f"\n\nThe object provided is from '{obj_module_name}', which is coming from {obj_location}.\nHowever, when we import '{obj2_module_name}', it's coming from {obj2_location}.\nTo fix this, make sure this 'PackageExporter's importer lists {obj_importer_name} before {obj2_importer_name}."
        raise ObjMismatchError(msg)

    def whichmodule(self, obj: Any, name: str) -> str:
        """Find the module name an object belongs to.

        This should be considered internal for end-users, but developers of
        an importer can override it to customize the behavior.

        Taken from pickle.py, but modified to exclude the search into sys.modules
        """
        module_name = getattr(obj, '__module__', None)
        if module_name is not None:
            return module_name
        for module_name, module in self.modules.copy().items():
            if module_name == '__main__' or module_name == '__mp_main__' or module is None:
                continue
            try:
                if _getattribute(module, name)[0] is obj:
                    return module_name
            except AttributeError:
                pass
        return '__main__'
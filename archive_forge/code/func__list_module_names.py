import collections
import importlib
import os
import pkgutil
def _list_module_names():
    module_names = []
    package_path = os.path.dirname(os.path.abspath(__file__))
    for _, module_name, ispkg in pkgutil.iter_modules(path=[package_path]):
        if module_name in IGNORED_MODULES or ispkg:
            continue
        else:
            module_names.append(module_name)
    return module_names
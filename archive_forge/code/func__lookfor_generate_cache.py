import sys
import os
import re
from numpy import ufunc
def _lookfor_generate_cache(module, import_modules, regenerate):
    """
    Generate docstring cache for given module.

    Parameters
    ----------
    module : str, None, module
        Module for which to generate docstring cache
    import_modules : bool
        Whether to import sub-modules in packages.
    regenerate : bool
        Re-generate the docstring cache

    Returns
    -------
    cache : dict {obj_full_name: (docstring, kind, index), ...}
        Docstring cache for the module, either cached one (regenerate=False)
        or newly generated.

    """
    import inspect
    from io import StringIO
    if module is None:
        module = 'numpy'
    if isinstance(module, str):
        try:
            __import__(module)
        except ImportError:
            return {}
        module = sys.modules[module]
    elif isinstance(module, list) or isinstance(module, tuple):
        cache = {}
        for mod in module:
            cache.update(_lookfor_generate_cache(mod, import_modules, regenerate))
        return cache
    if id(module) in _lookfor_caches and (not regenerate):
        return _lookfor_caches[id(module)]
    cache = {}
    _lookfor_caches[id(module)] = cache
    seen = {}
    index = 0
    stack = [(module.__name__, module)]
    while stack:
        name, item = stack.pop(0)
        if id(item) in seen:
            continue
        seen[id(item)] = True
        index += 1
        kind = 'object'
        if inspect.ismodule(item):
            kind = 'module'
            try:
                _all = item.__all__
            except AttributeError:
                _all = None
            if import_modules and hasattr(item, '__path__'):
                for pth in item.__path__:
                    for mod_path in os.listdir(pth):
                        this_py = os.path.join(pth, mod_path)
                        init_py = os.path.join(pth, mod_path, '__init__.py')
                        if os.path.isfile(this_py) and mod_path.endswith('.py'):
                            to_import = mod_path[:-3]
                        elif os.path.isfile(init_py):
                            to_import = mod_path
                        else:
                            continue
                        if to_import == '__init__':
                            continue
                        try:
                            old_stdout = sys.stdout
                            old_stderr = sys.stderr
                            try:
                                sys.stdout = StringIO()
                                sys.stderr = StringIO()
                                __import__('%s.%s' % (name, to_import))
                            finally:
                                sys.stdout = old_stdout
                                sys.stderr = old_stderr
                        except KeyboardInterrupt:
                            raise
                        except BaseException:
                            continue
            for n, v in _getmembers(item):
                try:
                    item_name = getattr(v, '__name__', '%s.%s' % (name, n))
                    mod_name = getattr(v, '__module__', None)
                except NameError:
                    item_name = '%s.%s' % (name, n)
                    mod_name = None
                if '.' not in item_name and mod_name:
                    item_name = '%s.%s' % (mod_name, item_name)
                if not item_name.startswith(name + '.'):
                    if isinstance(v, ufunc):
                        pass
                    else:
                        continue
                elif not (inspect.ismodule(v) or _all is None or n in _all):
                    continue
                stack.append(('%s.%s' % (name, n), v))
        elif inspect.isclass(item):
            kind = 'class'
            for n, v in _getmembers(item):
                stack.append(('%s.%s' % (name, n), v))
        elif hasattr(item, '__call__'):
            kind = 'func'
        try:
            doc = inspect.getdoc(item)
        except NameError:
            doc = None
        if doc is not None:
            cache[name] = (doc, kind, index)
    return cache
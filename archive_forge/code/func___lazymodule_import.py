def __lazymodule_import(self):
    """Import the module now."""
    local_name = self.__lazymodule_name
    full_name = self.__name__
    if self.__lazymodule_loaded:
        return self.__lazymodule_locals[local_name]
    if _debug:
        print('LazyModule: Loading module %r' % full_name)
    self.__lazymodule_locals[local_name] = module = __import__(full_name, self.__lazymodule_locals, self.__lazymodule_globals, '*')
    self.__dict__.update(module.__dict__)
    self.__dict__['__lazymodule_loaded'] = 1
    if _debug:
        print('LazyModule: Module %r loaded' % full_name)
    return module
def _find_and_load(name, import_):
    """Find and load the module."""
    module = sys.modules.get(name, _NEEDS_LOADING)
    if module is _NEEDS_LOADING or getattr(getattr(module, '__spec__', None), '_initializing', False):
        with _ModuleLockManager(name):
            module = sys.modules.get(name, _NEEDS_LOADING)
            if module is _NEEDS_LOADING:
                return _find_and_load_unlocked(name, import_)
        _lock_unlock_module(name)
    if module is None:
        message = 'import of {} halted; None in sys.modules'.format(name)
        raise ModuleNotFoundError(message, name=name)
    return module
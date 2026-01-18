def _load_unlocked(spec):
    if spec.loader is not None:
        if not hasattr(spec.loader, 'exec_module'):
            msg = f'{_object_name(spec.loader)}.exec_module() not found; falling back to load_module()'
            _warnings.warn(msg, ImportWarning)
            return _load_backward_compatible(spec)
    module = module_from_spec(spec)
    spec._initializing = True
    try:
        sys.modules[spec.name] = module
        try:
            if spec.loader is None:
                if spec.submodule_search_locations is None:
                    raise ImportError('missing loader', name=spec.name)
            else:
                spec.loader.exec_module(module)
        except:
            try:
                del sys.modules[spec.name]
            except KeyError:
                pass
            raise
        module = sys.modules.pop(spec.name)
        sys.modules[spec.name] = module
        _verbose_message('import {!r} # {!r}', spec.name, spec.loader)
    finally:
        spec._initializing = False
    return module
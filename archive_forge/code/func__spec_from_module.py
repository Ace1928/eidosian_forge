def _spec_from_module(module, loader=None, origin=None):
    try:
        spec = module.__spec__
    except AttributeError:
        pass
    else:
        if spec is not None:
            return spec
    name = module.__name__
    if loader is None:
        try:
            loader = module.__loader__
        except AttributeError:
            pass
    try:
        location = module.__file__
    except AttributeError:
        location = None
    if origin is None:
        if loader is not None:
            origin = getattr(loader, '_ORIGIN', None)
        if not origin and location is not None:
            origin = location
    try:
        cached = module.__cached__
    except AttributeError:
        cached = None
    try:
        submodule_search_locations = list(module.__path__)
    except AttributeError:
        submodule_search_locations = None
    spec = ModuleSpec(name, loader, origin=origin)
    spec._set_fileattr = False if location is None else origin == location
    spec.cached = cached
    spec.submodule_search_locations = submodule_search_locations
    return spec
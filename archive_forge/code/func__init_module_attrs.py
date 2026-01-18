def _init_module_attrs(spec, module, *, override=False):
    if override or getattr(module, '__name__', None) is None:
        try:
            module.__name__ = spec.name
        except AttributeError:
            pass
    if override or getattr(module, '__loader__', None) is None:
        loader = spec.loader
        if loader is None:
            if spec.submodule_search_locations is not None:
                if _bootstrap_external is None:
                    raise NotImplementedError
                NamespaceLoader = _bootstrap_external.NamespaceLoader
                loader = NamespaceLoader.__new__(NamespaceLoader)
                loader._path = spec.submodule_search_locations
                spec.loader = loader
                module.__file__ = None
        try:
            module.__loader__ = loader
        except AttributeError:
            pass
    if override or getattr(module, '__package__', None) is None:
        try:
            module.__package__ = spec.parent
        except AttributeError:
            pass
    try:
        module.__spec__ = spec
    except AttributeError:
        pass
    if override or getattr(module, '__path__', None) is None:
        if spec.submodule_search_locations is not None:
            try:
                module.__path__ = spec.submodule_search_locations
            except AttributeError:
                pass
    if spec.has_location:
        if override or getattr(module, '__file__', None) is None:
            try:
                module.__file__ = spec.origin
            except AttributeError:
                pass
        if override or getattr(module, '__cached__', None) is None:
            if spec.cached is not None:
                try:
                    module.__cached__ = spec.cached
                except AttributeError:
                    pass
    return module
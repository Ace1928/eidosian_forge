def _find_spec_legacy(finder, name, path):
    msg = f'{_object_name(finder)}.find_spec() not found; falling back to find_module()'
    _warnings.warn(msg, ImportWarning)
    loader = finder.find_module(name, path)
    if loader is None:
        return None
    return spec_from_loader(name, loader)
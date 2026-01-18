def _module_repr(module):
    """The implementation of ModuleType.__repr__()."""
    loader = getattr(module, '__loader__', None)
    if (spec := getattr(module, '__spec__', None)):
        return _module_repr_from_spec(spec)
    elif hasattr(loader, 'module_repr'):
        try:
            return loader.module_repr(module)
        except Exception:
            pass
    try:
        name = module.__name__
    except AttributeError:
        name = '?'
    try:
        filename = module.__file__
    except AttributeError:
        if loader is None:
            return '<module {!r}>'.format(name)
        else:
            return '<module {!r} ({!r})>'.format(name, loader)
    else:
        return '<module {!r} from {!r}>'.format(name, filename)
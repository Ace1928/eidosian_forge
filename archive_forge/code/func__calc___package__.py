def _calc___package__(globals):
    """Calculate what __package__ should be.

    __package__ is not guaranteed to be defined or could be set to None
    to represent that its proper value is unknown.

    """
    package = globals.get('__package__')
    spec = globals.get('__spec__')
    if package is not None:
        if spec is not None and package != spec.parent:
            _warnings.warn(f'__package__ != __spec__.parent ({package!r} != {spec.parent!r})', ImportWarning, stacklevel=3)
        return package
    elif spec is not None:
        return spec.parent
    else:
        _warnings.warn("can't resolve package from __spec__ or __package__, falling back on __name__ and __path__", ImportWarning, stacklevel=3)
        package = globals['__name__']
        if '__path__' not in globals:
            package = package.rpartition('.')[0]
    return package
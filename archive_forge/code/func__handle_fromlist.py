def _handle_fromlist(module, fromlist, import_, *, recursive=False):
    """Figure out what __import__ should return.

    The import_ parameter is a callable which takes the name of module to
    import. It is required to decouple the function from assuming importlib's
    import implementation is desired.

    """
    for x in fromlist:
        if not isinstance(x, str):
            if recursive:
                where = module.__name__ + '.__all__'
            else:
                where = "``from list''"
            raise TypeError(f'Item in {where} must be str, not {type(x).__name__}')
        elif x == '*':
            if not recursive and hasattr(module, '__all__'):
                _handle_fromlist(module, module.__all__, import_, recursive=True)
        elif not hasattr(module, x):
            from_name = '{}.{}'.format(module.__name__, x)
            try:
                _call_with_frames_removed(import_, from_name)
            except ModuleNotFoundError as exc:
                if exc.name == from_name and sys.modules.get(from_name, _NEEDS_LOADING) is not None:
                    continue
                raise
    return module
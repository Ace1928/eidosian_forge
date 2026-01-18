import sys
def _getcategory(category):
    if not category:
        return Warning
    if '.' not in category:
        import builtins as m
        klass = category
    else:
        module, _, klass = category.rpartition('.')
        try:
            m = __import__(module, None, None, [klass])
        except ImportError:
            raise _OptionError('invalid module name: %r' % (module,)) from None
    try:
        cat = getattr(m, klass)
    except AttributeError:
        raise _OptionError('unknown warning category: %r' % (category,)) from None
    if not issubclass(cat, Warning):
        raise _OptionError('invalid warning category: %r' % (category,))
    return cat
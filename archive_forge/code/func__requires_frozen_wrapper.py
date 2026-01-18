def _requires_frozen_wrapper(self, fullname):
    if not _imp.is_frozen(fullname):
        raise ImportError('{!r} is not a frozen module'.format(fullname), name=fullname)
    return fxn(self, fullname)
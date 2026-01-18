class AlreadyVersionedError(BzrError):
    """Used when a path is expected not to be versioned, but it is."""
    _fmt = '%(context_info)s%(path)s is already versioned.'

    def __init__(self, path, context_info=None):
        """Construct a new AlreadyVersionedError.

        :param path: This is the path which is versioned,
            which should be in a user friendly form.
        :param context_info: If given, this is information about the context,
            which could explain why this is expected to not be versioned.
        """
        BzrError.__init__(self)
        self.path = path
        if context_info is None:
            self.context_info = ''
        else:
            self.context_info = context_info + '. '
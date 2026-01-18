class NoGroupError(Error):
    """Raised when trying to access a nonexistant group in an INI-style file.
    
    Attributes are .group and .file.
    """

    def __init__(self, group, file):
        Error.__init__(self, 'No group: %s in file %s' % (group, file))
        self.group = group
        self.file = file
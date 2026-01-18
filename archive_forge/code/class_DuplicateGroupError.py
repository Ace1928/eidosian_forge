class DuplicateGroupError(Error):
    """Raised when the same key occurs twice in an INI-style file.
    
    Attributes are .group and .file.
    """

    def __init__(self, group, file):
        Error.__init__(self, 'Duplicate group: %s in file %s' % (group, file))
        self.group = group
        self.file = file
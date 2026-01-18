class NoKeyError(Error):
    """Raised when trying to access a nonexistant key in an INI-style file.
    
    Attributes are .key, .group and .file.
    """

    def __init__(self, key, group, file):
        Error.__init__(self, "No key '%s' in group %s of file %s" % (key, group, file))
        self.key = key
        self.group = group
        self.file = file
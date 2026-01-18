class UnknownDateFormat(ImportError):
    """Raised when an unknown date format is given."""
    _fmt = "Unknown date format '%(format)s'"

    def __init__(self, format):
        self.format = format
        ImportError.__init__(self)
class ParsingError(ImportError):
    """The base exception class for all import processing exceptions."""
    _fmt = _LOCATION_FMT + 'Unknown Import Parsing Error'

    def __init__(self, lineno):
        self.lineno = lineno
        ImportError.__init__(self)
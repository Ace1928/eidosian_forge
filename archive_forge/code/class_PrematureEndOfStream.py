class PrematureEndOfStream(ParsingError):
    """Raised when the 'done' feature was specified but missing."""
    _fmt = _LOCATION_FMT + "Stream end before 'done' command"

    def __init__(self, lineno):
        ParsingError.__init__(self, lineno)
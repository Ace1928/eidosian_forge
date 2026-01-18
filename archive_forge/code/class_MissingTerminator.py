class MissingTerminator(ParsingError):
    """Raised when EOF encountered while expecting to find a terminator."""
    _fmt = _LOCATION_FMT + "Unexpected EOF - expected '%(terminator)s' terminator"

    def __init__(self, lineno, terminator):
        self.terminator = terminator
        ParsingError.__init__(self, lineno)
class BadFormat(ParsingError):
    """Raised when a section is formatted incorrectly."""
    _fmt = _LOCATION_FMT + "Bad format for section %(section)s in command %(cmd)s: found '%(text)s'"

    def __init__(self, lineno, cmd, section, text):
        self.cmd = cmd
        self.section = section
        self.text = text
        ParsingError.__init__(self, lineno)
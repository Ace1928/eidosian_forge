class NonPrintableDefect(HeaderDefect):
    """ASCII characters outside the ascii-printable range found"""

    def __init__(self, non_printables):
        super().__init__(non_printables)
        self.non_printables = non_printables

    def __str__(self):
        return 'the following ASCII non-printables found in header: {}'.format(self.non_printables)
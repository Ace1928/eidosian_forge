class MessageDefect(ValueError):
    """Base class for a message defect."""

    def __init__(self, line=None):
        if line is not None:
            super().__init__(line)
        self.line = line
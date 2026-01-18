class wr:
    """WR lines."""

    def __init__(self, line=None):
        """Initialize the class."""
        self.name = ''
        self.aligned = ''
        self.program = ''
        self.date = []
        if line:
            header = line.split()
            self.name = header[0]
            self.aligned = header[1]
            self.program = header[2]
            self.date = header[3]
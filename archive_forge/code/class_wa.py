class wa:
    """WA (whole assembly tag), holds the assembly program name, version, etc."""

    def __init__(self, line=None):
        """Initialize the class."""
        self.tag_type = ''
        self.program = ''
        self.date = ''
        self.info = []
        if line:
            header = line.split()
            self.tag_type = header[0]
            self.program = header[1]
            self.date = header[2]
class ct:
    """CT (consensus tags)."""

    def __init__(self, line=None):
        """Initialize the class."""
        self.name = ''
        self.tag_type = ''
        self.program = ''
        self.padded_start = None
        self.padded_end = None
        self.date = ''
        self.notrans = ''
        self.info = []
        self.comment = []
        if line:
            header = line.split()
            self.name = header[0]
            self.tag_type = header[1]
            self.program = header[2]
            self.padded_start = int(header[3])
            self.padded_end = int(header[4])
            self.date = header[5]
            if len(header) == 7:
                self.notrans = header[6]
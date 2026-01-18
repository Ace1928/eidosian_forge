class Reads:
    """Holds information about a read supporting an ACE contig."""

    def __init__(self, line=None):
        """Initialize the class."""
        self.rd = None
        self.qa = None
        self.ds = None
        self.rt = None
        self.wr = None
        if line:
            self.rd = rd()
            header = line.split()
            self.rd.name = header[1]
            self.rd.padded_bases = int(header[2])
            self.rd.info_items = int(header[3])
            self.rd.read_tags = int(header[4])
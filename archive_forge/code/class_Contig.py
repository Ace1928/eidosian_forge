class Contig:
    """Holds information about a contig from an ACE record."""

    def __init__(self, line=None):
        """Initialize the class."""
        self.name = ''
        self.nbases = None
        self.nreads = None
        self.nsegments = None
        self.uorc = None
        self.sequence = ''
        self.quality = []
        self.af = []
        self.bs = []
        self.reads = []
        self.ct = None
        self.wa = None
        if line:
            header = line.split()
            self.name = header[1]
            self.nbases = int(header[2])
            self.nreads = int(header[3])
            self.nsegments = int(header[4])
            self.uorc = header[5]
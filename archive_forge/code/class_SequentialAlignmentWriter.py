class SequentialAlignmentWriter(AlignmentWriter):
    """Base class for building MultipleSeqAlignment writers.

    This assumes each alignment can be simply appended to the file.
    You should write a write_alignment() method.
    You may wish to redefine the __init__ method as well.
    """

    def __init__(self, handle):
        """Initialize the class."""
        self.handle = handle

    def write_file(self, alignments):
        """Use this to write an entire file containing the given alignments.

        Arguments:
         - alignments - A list or iterator returning MultipleSeqAlignment objects

        In general, this method can only be called once per file.
        """
        self.write_header()
        count = 0
        for alignment in alignments:
            self.write_alignment(alignment)
            count += 1
        self.write_footer()
        return count

    def write_header(self):
        """Use this to write any header.

        This method should be replaced by any derived class to do something
        useful.
        """

    def write_footer(self):
        """Use this to write any footer.

        This method should be replaced by any derived class to do something
        useful.
        """

    def write_alignment(self, alignment):
        """Use this to write a single alignment.

        This method should be replaced by any derived class to do something
        useful.
        """
        raise NotImplementedError('This object should be subclassed')
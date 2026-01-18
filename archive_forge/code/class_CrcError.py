class CrcError(ArchiveError):
    """Exception raised for CRC error when decompression.

    Attributes:
      expected -- expected CRC bytes
      actual -- actual CRC data
      filename -- filename that has CRC error
    """

    def __init__(self, expected, actual, filename):
        super().__init__(expected, actual, filename)
        self.expected = expected
        self.actual = actual
        self.filename = filename
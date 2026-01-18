import binascii
class ChecksumMismatch(Exception):
    """A checksum didn't match the expected contents."""

    def __init__(self, expected, got, extra=None) -> None:
        if len(expected) == 20:
            expected = binascii.hexlify(expected)
        if len(got) == 20:
            got = binascii.hexlify(got)
        self.expected = expected
        self.got = got
        self.extra = extra
        if self.extra is None:
            Exception.__init__(self, f'Checksum mismatch: Expected {expected}, got {got}')
        else:
            Exception.__init__(self, f'Checksum mismatch: Expected {expected}, got {got}; {extra}')
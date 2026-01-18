import binascii
class HangupException(GitProtocolError):
    """Hangup exception."""

    def __init__(self, stderr_lines=None) -> None:
        if stderr_lines:
            super().__init__('\n'.join([line.decode('utf-8', 'surrogateescape') for line in stderr_lines]))
        else:
            super().__init__('The remote server unexpectedly closed the connection.')
        self.stderr_lines = stderr_lines

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.stderr_lines == other.stderr_lines
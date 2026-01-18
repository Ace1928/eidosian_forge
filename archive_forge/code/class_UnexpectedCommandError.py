import binascii
class UnexpectedCommandError(GitProtocolError):
    """Unexpected command received in a proto line."""

    def __init__(self, command) -> None:
        if command is None:
            command = 'flush-pkt'
        else:
            command = 'command %s' % command
        super().__init__('Protocol got unexpected %s' % command)
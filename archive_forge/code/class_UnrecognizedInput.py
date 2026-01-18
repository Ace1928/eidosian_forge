class UnrecognizedInput(PlexError):
    scanner = None
    position = None
    state_name = None

    def __init__(self, scanner, state_name):
        self.scanner = scanner
        self.position = scanner.get_position()
        self.state_name = state_name

    def __str__(self):
        return "'%s', line %d, char %d: Token not recognised in state %r" % (self.position + (self.state_name,))
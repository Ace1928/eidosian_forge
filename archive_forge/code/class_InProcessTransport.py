class InProcessTransport(BzrError):
    _fmt = "The transport '%(transport)s' is only accessible within this process."

    def __init__(self, transport):
        self.transport = transport
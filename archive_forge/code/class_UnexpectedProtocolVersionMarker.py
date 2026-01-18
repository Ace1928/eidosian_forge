class UnexpectedProtocolVersionMarker(TransportError):
    _fmt = 'Received bad protocol version marker: %(marker)r'

    def __init__(self, marker):
        self.marker = marker
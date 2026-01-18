class ConnectionReset(TransportError):
    _fmt = 'Connection closed: %(msg)s %(orig_error)s'
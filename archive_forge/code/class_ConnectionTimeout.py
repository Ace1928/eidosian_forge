class ConnectionTimeout(ConnectionError):
    _fmt = 'Connection Timeout: %(msg)s%(orig_error)s'
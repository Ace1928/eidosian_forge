from twisted.internet.endpoints import _parse
def _parseTCPSSL(factory, domain, port):
    """For the moment, parse TCP or SSL connections the same"""
    return ((domain, int(port), factory), {})
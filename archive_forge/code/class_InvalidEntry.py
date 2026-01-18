from twisted.cred.error import UnauthorizedLogin
class InvalidEntry(Exception):
    """
    An entry in a known_hosts file could not be interpreted as a valid entry.
    """
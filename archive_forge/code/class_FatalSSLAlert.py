import errno
import sys
class FatalSSLAlert(Exception):
    """Exception raised when the SSL implementation signals a fatal alert."""
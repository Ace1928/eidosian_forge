import socket
class PasswordRequiredException(AuthenticationException):
    """
    Exception raised when a password is needed to unlock a private key file.
    """
    pass
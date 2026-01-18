import socket
class BadAuthenticationType(AuthenticationException):
    """
    Exception raised when an authentication type (like password) is used, but
    the server isn't allowing that type.  (It may only allow public-key, for
    example.)

    .. versionadded:: 1.1
    """
    allowed_types = []

    def __init__(self, explanation, types):
        AuthenticationException.__init__(self, explanation, types)
        self.explanation = explanation
        self.allowed_types = types

    def __str__(self):
        return '{}; allowed types: {!r}'.format(self.explanation, self.allowed_types)
import socket
class ProxyCommandFailure(SSHException):
    """
    The "ProxyCommand" found in the .ssh/config file returned an error.

    :param str command: The command line that is generating this exception.
    :param str error: The error captured from the proxy command output.
    """

    def __init__(self, command, error):
        SSHException.__init__(self, command, error)
        self.command = command
        self.error = error

    def __str__(self):
        return 'ProxyCommand("{}") returned nonzero exit status: {}'.format(self.command, self.error)
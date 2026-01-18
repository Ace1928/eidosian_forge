from typing import Optional
class NoSupportedAuthentication(IMAP4Exception):

    def __init__(self, serverSupports, clientSupports):
        IMAP4Exception.__init__(self, 'No supported authentication schemes available')
        self.serverSupports = serverSupports
        self.clientSupports = clientSupports

    def __str__(self) -> str:
        return IMAP4Exception.__str__(self) + ': Server supports {!r}, client supports {!r}'.format(self.serverSupports, self.clientSupports)
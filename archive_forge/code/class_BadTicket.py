import time as time_mod
from http.cookies import SimpleCookie
from urllib.parse import quote as url_quote
from urllib.parse import unquote as url_unquote
from paste import request
class BadTicket(Exception):
    """
    Exception raised when a ticket can't be parsed.  If we get
    far enough to determine what the expected digest should have
    been, expected is set.  This should not be shown by default,
    but can be useful for debugging.
    """

    def __init__(self, msg, expected=None):
        self.expected = expected
        Exception.__init__(self, msg)
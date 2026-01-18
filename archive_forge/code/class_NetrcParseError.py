import os, shlex, stat
class NetrcParseError(Exception):
    """Exception raised on syntax errors in the .netrc file."""

    def __init__(self, msg, filename=None, lineno=None):
        self.filename = filename
        self.lineno = lineno
        self.msg = msg
        Exception.__init__(self, msg)

    def __str__(self):
        return '%s (%s, line %s)' % (self.msg, self.filename, self.lineno)
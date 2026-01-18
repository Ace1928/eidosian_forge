import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import (
from twisted.protocols import basic, policies
from twisted.python import log
def _dotUnquoter(line):
    """
    Remove a byte-stuffed termination character at the beginning of a line if
    present.

    When the termination character (C{'.'}) appears at the beginning of a line,
    the server byte-stuffs it by adding another termination character to
    avoid confusion with the terminating sequence (C{'.\\r\\n'}).

    @type line: L{bytes}
    @param line: A received line.

    @rtype: L{bytes}
    @return: The line without the byte-stuffed termination character at the
        beginning if it was present. Otherwise, the line unchanged.
    """
    if line.startswith(b'..'):
        return line[1:]
    return line
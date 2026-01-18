import os
import sys
from zope.interface import implementer
import pywintypes
import win32api
import win32con
import win32event
import win32file
import win32pipe
import win32process
import win32security
from twisted.internet import _pollingfile, error
from twisted.internet._baseprocess import BaseProcess
from twisted.internet.interfaces import IConsumer, IProcessTransport, IProducer
from twisted.python.win32 import quoteArguments
def _findShebang(filename):
    """
    Look for a #! line, and return the value following the #! if one exists, or
    None if this file is not a script.

    I don't know if there are any conventions for quoting in Windows shebang
    lines, so this doesn't support any; therefore, you may not pass any
    arguments to scripts invoked as filters.  That's probably wrong, so if
    somebody knows more about the cultural expectations on Windows, please feel
    free to fix.

    This shebang line support was added in support of the CGI tests;
    appropriately enough, I determined that shebang lines are culturally
    accepted in the Windows world through this page::

        http://www.cgi101.com/learn/connect/winxp.html

    @param filename: str representing a filename

    @return: a str representing another filename.
    """
    with open(filename) as f:
        if f.read(2) == '#!':
            exe = f.readline(1024).strip('\n')
            return exe
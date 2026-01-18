import os
import sys
import termios
import tty
from twisted.conch.insults.insults import ServerProtocol
from twisted.conch.manhole import ColoredManhole
from twisted.internet import defer, protocol, reactor, stdio
from twisted.python import failure, log, reflect
class UnexpectedOutputError(Exception):
    pass
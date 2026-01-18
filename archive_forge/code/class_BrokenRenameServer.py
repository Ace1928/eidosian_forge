import errno
import socket
import socketserver
import sys
import threading
from breezy import cethread, errors, osutils, transport, urlutils
from breezy.bzr.smart import medium, server
from breezy.transport import chroot, pathfilter
class BrokenRenameServer(DecoratorServer):
    """Server for the BrokenRenameTransportDecorator for testing with."""

    def get_decorator_class(self):
        from breezy.transport import brokenrename
        return brokenrename.BrokenRenameTransportDecorator
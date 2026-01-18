import socket
from incremental import Version
from twisted.python import deprecate
class FileDescriptorOverrun(ConnectionLost):
    """
    A mis-use of L{IUNIXTransport.sendFileDescriptor} caused the connection to
    be closed.

    Each file descriptor sent using C{sendFileDescriptor} must be associated
    with at least one byte sent using L{ITransport.write}.  If at any point
    fewer bytes have been written than file descriptors have been sent, the
    connection is closed with this exception.
    """
    MESSAGE = 'A mis-use of IUNIXTransport.sendFileDescriptor caused the connection to be closed.'
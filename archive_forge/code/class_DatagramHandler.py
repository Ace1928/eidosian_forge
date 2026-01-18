import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy
class DatagramHandler(SocketHandler):
    """
    A handler class which writes logging records, in pickle format, to
    a datagram socket.  The pickle which is sent is that of the LogRecord's
    attribute dictionary (__dict__), so that the receiver does not need to
    have the logging module installed in order to process the logging event.

    To unpickle the record at the receiving end into a LogRecord, use the
    makeLogRecord function.

    """

    def __init__(self, host, port):
        """
        Initializes the handler with a specific host address and port.
        """
        SocketHandler.__init__(self, host, port)
        self.closeOnError = False

    def makeSocket(self):
        """
        The factory method of SocketHandler is here overridden to create
        a UDP socket (SOCK_DGRAM).
        """
        if self.port is None:
            family = socket.AF_UNIX
        else:
            family = socket.AF_INET
        s = socket.socket(family, socket.SOCK_DGRAM)
        return s

    def send(self, s):
        """
        Send a pickled string to a socket.

        This function no longer allows for partial sends which can happen
        when the network is busy - UDP does not guarantee delivery and
        can deliver packets out of sequence.
        """
        if self.sock is None:
            self.createSocket()
        self.sock.sendto(s, self.address)
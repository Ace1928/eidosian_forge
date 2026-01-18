from __future__ import absolute_import
import errno
import fcntl
import os
import select
import struct
import sys
import termios
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
class VTIMESerial(Serial):
    """    Implement timeout using vtime of tty device instead of using select.
    This means that no inter character timeout can be specified and that
    the error handling is degraded.

    Overall timeout is disabled when inter-character timeout is used.

    Note that this implementation does NOT support cancel_read(), it will
    just ignore that.
    """

    def _reconfigure_port(self, force_update=True):
        """Set communication parameters on opened port."""
        super(VTIMESerial, self)._reconfigure_port()
        fcntl.fcntl(self.fd, fcntl.F_SETFL, 0)
        if self._inter_byte_timeout is not None:
            vmin = 1
            vtime = int(self._inter_byte_timeout * 10)
        elif self._timeout is None:
            vmin = 1
            vtime = 0
        else:
            vmin = 0
            vtime = int(self._timeout * 10)
        try:
            orig_attr = termios.tcgetattr(self.fd)
            iflag, oflag, cflag, lflag, ispeed, ospeed, cc = orig_attr
        except termios.error as msg:
            raise serial.SerialException('Could not configure port: {}'.format(msg))
        if vtime < 0 or vtime > 255:
            raise ValueError('Invalid vtime: {!r}'.format(vtime))
        cc[termios.VTIME] = vtime
        cc[termios.VMIN] = vmin
        termios.tcsetattr(self.fd, termios.TCSANOW, [iflag, oflag, cflag, lflag, ispeed, ospeed, cc])

    def read(self, size=1):
        """        Read size bytes from the serial port. If a timeout is set it may
        return less characters as requested. With no timeout it will block
        until the requested number of bytes is read.
        """
        if not self.is_open:
            raise PortNotOpenError()
        read = bytearray()
        while len(read) < size:
            buf = os.read(self.fd, size - len(read))
            if not buf:
                break
            read.extend(buf)
        return bytes(read)
    cancel_read = property()
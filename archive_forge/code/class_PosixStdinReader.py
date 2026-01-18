from __future__ import unicode_literals
from codecs import getincrementaldecoder
import os
import six
class PosixStdinReader(object):
    """
    Wrapper around stdin which reads (nonblocking) the next available 1024
    bytes and decodes it.

    Note that you can't be sure that the input file is closed if the ``read``
    function returns an empty string. When ``errors=ignore`` is passed,
    ``read`` can return an empty string if all malformed input was replaced by
    an empty string. (We can't block here and wait for more input.) So, because
    of that, check the ``closed`` attribute, to be sure that the file has been
    closed.

    :param stdin_fd: File descriptor from which we read.
    :param errors:  Can be 'ignore', 'strict' or 'replace'.
        On Python3, this can be 'surrogateescape', which is the default.

        'surrogateescape' is preferred, because this allows us to transfer
        unrecognised bytes to the key bindings. Some terminals, like lxterminal
        and Guake, use the 'Mxx' notation to send mouse events, where each 'x'
        can be any possible byte.
    """

    def __init__(self, stdin_fd, errors='ignore' if six.PY2 else 'surrogateescape'):
        assert isinstance(stdin_fd, int)
        self.stdin_fd = stdin_fd
        self.errors = errors
        self._stdin_decoder_cls = getincrementaldecoder('utf-8')
        self._stdin_decoder = self._stdin_decoder_cls(errors=errors)
        self.closed = False

    def read(self, count=1024):
        """
        Read the input and return it as a string.

        Return the text. Note that this can return an empty string, even when
        the input stream was not yet closed. This means that something went
        wrong during the decoding.
        """
        if self.closed:
            return b''
        try:
            data = os.read(self.stdin_fd, count)
            if data == b'':
                self.closed = True
                return ''
        except OSError:
            data = b''
        return self._stdin_decoder.decode(data)
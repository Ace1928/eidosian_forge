import base64
import binascii
import collections
import html
from django.conf import settings
from django.core.exceptions import (
from django.core.files.uploadhandler import SkipFile, StopFutureHandlers, StopUpload
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.http import parse_header_parameters
from django.utils.regex_helper import _lazy_re_compile
class LazyStream:
    """
    The LazyStream wrapper allows one to get and "unget" bytes from a stream.

    Given a producer object (an iterator that yields bytestrings), the
    LazyStream object will support iteration, reading, and keeping a "look-back"
    variable in case you need to "unget" some bytes.
    """

    def __init__(self, producer, length=None):
        """
        Every LazyStream must have a producer when instantiated.

        A producer is an iterable that returns a string each time it
        is called.
        """
        self._producer = producer
        self._empty = False
        self._leftover = b''
        self.length = length
        self.position = 0
        self._remaining = length
        self._unget_history = []

    def tell(self):
        return self.position

    def read(self, size=None):

        def parts():
            remaining = self._remaining if size is None else size
            if remaining is None:
                yield b''.join(self)
                return
            while remaining != 0:
                assert remaining > 0, 'remaining bytes to read should never go negative'
                try:
                    chunk = next(self)
                except StopIteration:
                    return
                else:
                    emitting = chunk[:remaining]
                    self.unget(chunk[remaining:])
                    remaining -= len(emitting)
                    yield emitting
        return b''.join(parts())

    def __next__(self):
        """
        Used when the exact number of bytes to read is unimportant.

        Return whatever chunk is conveniently returned from the iterator.
        Useful to avoid unnecessary bookkeeping if performance is an issue.
        """
        if self._leftover:
            output = self._leftover
            self._leftover = b''
        else:
            output = next(self._producer)
            self._unget_history = []
        self.position += len(output)
        return output

    def close(self):
        """
        Used to invalidate/disable this lazy stream.

        Replace the producer with an empty list. Any leftover bytes that have
        already been read will still be reported upon read() and/or next().
        """
        self._producer = []

    def __iter__(self):
        return self

    def unget(self, bytes):
        """
        Place bytes back onto the front of the lazy stream.

        Future calls to read() will return those bytes first. The
        stream position and thus tell() will be rewound.
        """
        if not bytes:
            return
        self._update_unget_history(len(bytes))
        self.position -= len(bytes)
        self._leftover = bytes + self._leftover

    def _update_unget_history(self, num_bytes):
        """
        Update the unget history as a sanity check to see if we've pushed
        back the same number of bytes in one chunk. If we keep ungetting the
        same number of bytes many times (here, 50), we're mostly likely in an
        infinite loop of some sort. This is usually caused by a
        maliciously-malformed MIME request.
        """
        self._unget_history = [num_bytes] + self._unget_history[:49]
        number_equal = len([current_number for current_number in self._unget_history if current_number == num_bytes])
        if number_equal > 40:
            raise SuspiciousMultipartForm("The multipart parser got stuck, which shouldn't happen with normal uploaded files. Check for malicious upload activity; if there is none, report this to the Django developers.")
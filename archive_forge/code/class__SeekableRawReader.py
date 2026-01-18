import io
import functools
import logging
import time
import warnings
import smart_open.bytebuffer
import smart_open.concurrency
import smart_open.utils
from smart_open import constants
class _SeekableRawReader(object):
    """Read an S3 object.

    This class is internal to the S3 submodule.
    """

    def __init__(self, client, bucket, key, version_id=None):
        self._client = client
        self._bucket = bucket
        self._key = key
        self._version_id = version_id
        self._content_length = None
        self._position = 0
        self._body = None

    def seek(self, offset, whence=constants.WHENCE_START):
        """Seek to the specified position.

        :param int offset: The offset in bytes.
        :param int whence: Where the offset is from.

        :returns: the position after seeking.
        :rtype: int
        """
        if whence not in constants.WHENCE_CHOICES:
            raise ValueError('invalid whence, expected one of %r' % constants.WHENCE_CHOICES)
        if self._body is not None:
            self._body.close()
        self._body = None
        start = None
        stop = None
        if whence == constants.WHENCE_START:
            start = max(0, offset)
        elif whence == constants.WHENCE_CURRENT:
            start = max(0, offset + self._position)
        else:
            stop = max(0, -offset)
        if self._content_length is None:
            reached_eof = False
        elif start is not None and start >= self._content_length:
            reached_eof = True
        elif stop == 0:
            reached_eof = True
        else:
            reached_eof = False
        if reached_eof:
            self._body = io.BytesIO()
            self._position = self._content_length
        else:
            self._open_body(start, stop)
        return self._position

    def _open_body(self, start=None, stop=None):
        """Open a connection to download the specified range of bytes. Store
        the open file handle in self._body.

        If no range is specified, start defaults to self._position.
        start and stop follow the semantics of the http range header,
        so a stop without a start will read bytes beginning at stop.

        As a side effect, set self._content_length. Set self._position
        to self._content_length if start is past end of file.
        """
        if start is None and stop is None:
            start = self._position
        range_string = smart_open.utils.make_range_string(start, stop)
        try:
            response = _get(self._client, self._bucket, self._key, self._version_id, range_string)
        except IOError as ioe:
            error_response = _unwrap_ioerror(ioe)
            if error_response is None or error_response.get('Code') != _OUT_OF_RANGE:
                raise
            try:
                self._position = self._content_length = int(error_response['ActualObjectSize'])
                self._body = io.BytesIO()
            except KeyError:
                response = _get(self._client, self._bucket, self._key, self._version_id, None)
                self._position = self._content_length = response['ContentLength']
                self._body = response['Body']
        else:
            logger.debug('%s: RetryAttempts: %d', self, response['ResponseMetadata']['RetryAttempts'])
            _, start, stop, length = smart_open.utils.parse_content_range(response['ContentRange'])
            self._content_length = length
            self._position = start
            self._body = response['Body']

    def read(self, size=-1):
        """Read from the continuous connection with the remote peer."""
        if self._body is None:
            self._open_body()
        if self._position >= self._content_length:
            return b''
        for attempt, seconds in enumerate([1, 2, 4, 8, 16], 1):
            try:
                if size == -1:
                    binary = self._body.read()
                else:
                    binary = self._body.read(size)
            except (ConnectionResetError, botocore.exceptions.BotoCoreError, urllib3.exceptions.HTTPError) as err:
                logger.warning('%s: caught %r while reading %d bytes, sleeping %ds before retry', self, err, size, seconds)
                time.sleep(seconds)
                self._open_body()
            else:
                self._position += len(binary)
                return binary
        raise IOError('%s: failed to read %d bytes after %d attempts' % (self, size, attempt))

    def __str__(self):
        return 'smart_open.s3._SeekableReader(%r, %r)' % (self._bucket, self._key)
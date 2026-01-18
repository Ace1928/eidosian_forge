import io
import functools
import logging
import time
import warnings
import smart_open.bytebuffer
import smart_open.concurrency
import smart_open.utils
from smart_open import constants
class SinglepartWriter(io.BufferedIOBase):
    """Writes bytes to S3 using the single part API.

    Implements the io.BufferedIOBase interface of the standard library.

    This class buffers all of its input in memory until its `close` method is called. Only then will
    the data be written to S3 and the buffer is released."""

    def __init__(self, bucket, key, client=None, client_kwargs=None, writebuffer=None):
        _initialize_boto3(self, client, client_kwargs, bucket, key)
        try:
            self._client.head_bucket(Bucket=bucket)
        except botocore.client.ClientError as e:
            raise ValueError('the bucket %r does not exist, or is forbidden for access' % bucket) from e
        if writebuffer is None:
            self._buf = io.BytesIO()
        else:
            self._buf = writebuffer
        self._total_bytes = 0
        self.raw = None

    def flush(self):
        pass

    def close(self):
        if self._buf is None:
            return
        self._buf.seek(0)
        try:
            self._client.put_object(Bucket=self._bucket, Key=self._key, Body=self._buf)
        except botocore.client.ClientError as e:
            raise ValueError('the bucket %r does not exist, or is forbidden for access' % self._bucket) from e
        logger.debug('%s: direct upload finished', self)
        self._buf = None

    @property
    def closed(self):
        return self._buf is None

    def writable(self):
        """Return True if the stream supports writing."""
        return True

    def seekable(self):
        """If False, seek(), tell() and truncate() will raise IOError.

        We offer only tell support, and no seek or truncate support."""
        return True

    def seek(self, offset, whence=constants.WHENCE_START):
        """Unsupported."""
        raise io.UnsupportedOperation

    def truncate(self, size=None):
        """Unsupported."""
        raise io.UnsupportedOperation

    def tell(self):
        """Return the current stream position."""
        return self._total_bytes

    def detach(self):
        raise io.UnsupportedOperation('detach() not supported')

    def write(self, b):
        """Write the given buffer (bytes, bytearray, memoryview or any buffer
        interface implementation) into the buffer. Content of the buffer will be
        written to S3 on close as a single-part upload.

        For more information about buffers, see https://docs.python.org/3/c-api/buffer.html"""
        length = self._buf.write(b)
        self._total_bytes += length
        return length

    def terminate(self):
        """Nothing to cancel in single-part uploads."""
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.terminate()
        else:
            self.close()

    def __str__(self):
        return 'smart_open.s3.SinglepartWriter(%r, %r)' % (self._object.bucket_name, self._object.key)

    def __repr__(self):
        return 'smart_open.s3.SinglepartWriter(bucket=%r, key=%r)' % (self._bucket, self._key)
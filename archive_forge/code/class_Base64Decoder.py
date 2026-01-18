import base64
import binascii
from .exceptions import DecodeError
class Base64Decoder:
    """This object provides an interface to decode a stream of Base64 data.  It
    is instantiated with an "underlying object", and whenever a write()
    operation is performed, it will decode the incoming data as Base64, and
    call write() on the underlying object.  This is primarily used for decoding
    form data encoded as Base64, but can be used for other purposes::

        from multipart.decoders import Base64Decoder
        fd = open("notb64.txt", "wb")
        decoder = Base64Decoder(fd)
        try:
            decoder.write("Zm9vYmFy")       # "foobar" in Base64
            decoder.finalize()
        finally:
            decoder.close()

        # The contents of "notb64.txt" should be "foobar".

    This object will also pass all finalize() and close() calls to the
    underlying object, if the underlying object supports them.

    Note that this class maintains a cache of base64 chunks, so that a write of
    arbitrary size can be performed.  You must call :meth:`finalize` on this
    object after all writes are completed to ensure that all data is flushed
    to the underlying object.

    :param underlying: the underlying object to pass writes to
    """

    def __init__(self, underlying):
        self.cache = bytearray()
        self.underlying = underlying

    def write(self, data):
        """Takes any input data provided, decodes it as base64, and passes it
        on to the underlying object.  If the data provided is invalid base64
        data, then this method will raise
        a :class:`multipart.exceptions.DecodeError`

        :param data: base64 data to decode
        """
        if len(self.cache) > 0:
            data = self.cache + data
        decode_len = len(data) // 4 * 4
        val = data[:decode_len]
        if len(val) > 0:
            try:
                decoded = base64.b64decode(val)
            except binascii.Error:
                raise DecodeError('There was an error raised while decoding base64-encoded data.')
            self.underlying.write(decoded)
        remaining_len = len(data) % 4
        if remaining_len > 0:
            self.cache = data[-remaining_len:]
        else:
            self.cache = b''
        return len(data)

    def close(self):
        """Close this decoder.  If the underlying object has a `close()`
        method, this function will call it.
        """
        if hasattr(self.underlying, 'close'):
            self.underlying.close()

    def finalize(self):
        """Finalize this object.  This should be called when no more data
        should be written to the stream.  This function can raise a
        :class:`multipart.exceptions.DecodeError` if there is some remaining
        data in the cache.

        If the underlying object has a `finalize()` method, this function will
        call it.
        """
        if len(self.cache) > 0:
            raise DecodeError('There are %d bytes remaining in the Base64Decoder cache when finalize() is called' % len(self.cache))
        if hasattr(self.underlying, 'finalize'):
            self.underlying.finalize()

    def __repr__(self):
        return f'{self.__class__.__name__}(underlying={self.underlying!r})'
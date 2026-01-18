import base64
import binascii
from .exceptions import DecodeError
class QuotedPrintableDecoder:
    """This object provides an interface to decode a stream of quoted-printable
    data.  It is instantiated with an "underlying object", in the same manner
    as the :class:`multipart.decoders.Base64Decoder` class.  This class behaves
    in exactly the same way, including maintaining a cache of quoted-printable
    chunks.

    :param underlying: the underlying object to pass writes to
    """

    def __init__(self, underlying):
        self.cache = b''
        self.underlying = underlying

    def write(self, data):
        """Takes any input data provided, decodes it as quoted-printable, and
        passes it on to the underlying object.

        :param data: quoted-printable data to decode
        """
        if len(self.cache) > 0:
            data = self.cache + data
        if data[-2:].find(b'=') != -1:
            enc, rest = (data[:-2], data[-2:])
        else:
            enc = data
            rest = b''
        if len(enc) > 0:
            self.underlying.write(binascii.a2b_qp(enc))
        self.cache = rest
        return len(data)

    def close(self):
        """Close this decoder.  If the underlying object has a `close()`
        method, this function will call it.
        """
        if hasattr(self.underlying, 'close'):
            self.underlying.close()

    def finalize(self):
        """Finalize this object.  This should be called when no more data
        should be written to the stream.  This function will not raise any
        exceptions, but it may write more data to the underlying object if
        there is data remaining in the cache.

        If the underlying object has a `finalize()` method, this function will
        call it.
        """
        if len(self.cache) > 0:
            self.underlying.write(binascii.a2b_qp(self.cache))
            self.cache = b''
        if hasattr(self.underlying, 'finalize'):
            self.underlying.finalize()

    def __repr__(self):
        return f'{self.__class__.__name__}(underlying={self.underlying!r})'
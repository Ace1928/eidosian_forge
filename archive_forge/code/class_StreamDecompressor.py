from __future__ import absolute_import
import struct
import cramjam
class StreamDecompressor:
    """This class implements the decompressor-side of the proposed Snappy
    framing format, found at

        http://code.google.com/p/snappy/source/browse/trunk/framing_format.txt
            ?spec=svn68&r=71

    This class matches a subset of the interface found for the zlib module's
    decompression objects (see zlib.decompressobj). Specifically, it currently
    implements the decompress method without the max_length option, the flush
    method without the length option, and the copy method.
    """

    def __init__(self):
        self.c = cramjam.snappy.Decompressor()
        self.remains = None

    @staticmethod
    def check_format(data):
        """Checks that the given data starts with snappy framing format
        stream identifier.
        Raises UncompressError if it doesn't start with the identifier.
        :return: None
        """
        if len(data) < 6:
            raise UncompressError('Too short data length')
        chunk_type = struct.unpack('<L', data[:4])[0]
        size = chunk_type >> 8
        chunk_type &= 255
        if chunk_type != _IDENTIFIER_CHUNK or size != len(_STREAM_IDENTIFIER):
            raise UncompressError('stream missing snappy identifier')
        chunk = data[4:4 + size]
        if chunk != _STREAM_IDENTIFIER:
            raise UncompressError('stream has invalid snappy identifier')

    def decompress(self, data: bytes):
        """Decompress 'data', returning a string containing the uncompressed
        data corresponding to at least part of the data in string. This data
        should be concatenated to the output produced by any preceding calls to
        the decompress() method. Some of the input data may be preserved in
        internal buffers for later processing.
        """
        if self.remains:
            data = self.remains + data
            self.remains = None
        if not data.startswith(_STREAM_HEADER_BLOCK):
            data = _STREAM_HEADER_BLOCK + data
        ldata = len(data)
        bsize = len(_STREAM_HEADER_BLOCK)
        if bsize + 4 > ldata:
            self.remains = data
            return b''
        while True:
            this_size = int.from_bytes(data[bsize + 1:bsize + 4], 'little') + 4
            if bsize == ldata:
                break
            if this_size + bsize > ldata:
                self.remains = data[bsize:]
                data = data[:bsize]
                break
            bsize += this_size
        self.c.decompress(data)
        return self.flush()

    def flush(self):
        return bytes(self.c.flush())

    def copy(self):
        return self
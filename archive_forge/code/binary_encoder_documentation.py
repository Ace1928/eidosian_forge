from struct import pack
from binascii import crc32
Encoder for the avro binary format.

    NOTE: All attributes and methods on this class should be considered
    private.

    Parameters
    ----------
    fo: file-like
        Input stream

    
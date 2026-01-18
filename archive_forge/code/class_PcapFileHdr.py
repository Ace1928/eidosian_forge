import struct
import sys
import time
class PcapFileHdr(object):
    """
    Global Header
    typedef struct pcap_hdr_s {
                guint32 magic_number;   /* magic number */
                guint16 version_major;  /* major version number */
                guint16 version_minor;  /* minor version number */
                gint32  thiszone;       /* GMT to local correction */
                guint32 sigfigs;        /* accuracy of timestamps */
                guint32 snaplen;        /* max length of captured packets,
                                           in octets */
                guint32 network;        /* data link type */
    } pcap_hdr_t;

    0                   1                   2                   3
    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                          Magic Number                         |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |        Version Major          |        Version Minor          |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                            Thiszone                           |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                            Sigfigs                            |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                            Snaplen                            |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                            Network                            |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
                                File Format
    """
    _FILE_HDR_FMT = '4sHHIIII'
    _FILE_HDR_FMT_BIG_ENDIAN = '>' + _FILE_HDR_FMT
    _FILE_HDR_FMT_LITTLE_ENDIAN = '<' + _FILE_HDR_FMT
    FILE_HDR_SIZE = struct.calcsize(_FILE_HDR_FMT)
    MAGIC_NUMBER_IDENTICAL = b'\xa1\xb2\xc3\xd4'
    MAGIC_NUMBER_SWAPPED = b'\xd4\xc3\xb2\xa1'

    def __init__(self, magic=MAGIC_NUMBER_SWAPPED, version_major=2, version_minor=4, thiszone=0, sigfigs=0, snaplen=0, network=0):
        self.magic = magic
        self.version_major = version_major
        self.version_minor = version_minor
        self.thiszone = thiszone
        self.sigfigs = sigfigs
        self.snaplen = snaplen
        self.network = network

    @classmethod
    def parser(cls, buf):
        magic_buf = buf[:4]
        if magic_buf == cls.MAGIC_NUMBER_IDENTICAL:
            fmt = cls._FILE_HDR_FMT_BIG_ENDIAN
            byteorder = 'big'
        elif magic_buf == cls.MAGIC_NUMBER_SWAPPED:
            fmt = cls._FILE_HDR_FMT_LITTLE_ENDIAN
            byteorder = 'little'
        else:
            raise struct.error('Invalid byte ordered pcap file.')
        return (cls(*struct.unpack_from(fmt, buf)), byteorder)

    def serialize(self):
        if sys.byteorder == 'big':
            fmt = self._FILE_HDR_FMT_BIG_ENDIAN
            self.magic = self.MAGIC_NUMBER_IDENTICAL
        else:
            fmt = self._FILE_HDR_FMT_LITTLE_ENDIAN
            self.magic = self.MAGIC_NUMBER_SWAPPED
        return struct.pack(fmt, self.magic, self.version_major, self.version_minor, self.thiszone, self.sigfigs, self.snaplen, self.network)
import struct
import sys
import zlib
from builtins import open as _open
def BgzfBlocks(handle):
    """Low level debugging function to inspect BGZF blocks.

    Expects a BGZF compressed file opened in binary read mode using
    the builtin open function. Do not use a handle from this bgzf
    module or the gzip module's open function which will decompress
    the file.

    Returns the block start offset (see virtual offsets), the block
    length (add these for the start of the next block), and the
    decompressed length of the blocks contents (limited to 65536 in
    BGZF), as an iterator - one tuple per BGZF block.

    >>> from builtins import open
    >>> handle = open("SamBam/ex1.bam", "rb")
    >>> for values in BgzfBlocks(handle):
    ...     print("Raw start %i, raw length %i; data start %i, data length %i" % values)
    Raw start 0, raw length 18239; data start 0, data length 65536
    Raw start 18239, raw length 18223; data start 65536, data length 65536
    Raw start 36462, raw length 18017; data start 131072, data length 65536
    Raw start 54479, raw length 17342; data start 196608, data length 65536
    Raw start 71821, raw length 17715; data start 262144, data length 65536
    Raw start 89536, raw length 17728; data start 327680, data length 65536
    Raw start 107264, raw length 17292; data start 393216, data length 63398
    Raw start 124556, raw length 28; data start 456614, data length 0
    >>> handle.close()

    Indirectly we can tell this file came from an old version of
    samtools because all the blocks (except the final one and the
    dummy empty EOF marker block) are 65536 bytes.  Later versions
    avoid splitting a read between two blocks, and give the header
    its own block (useful to speed up replacing the header). You
    can see this in ex1_refresh.bam created using samtools 0.1.18:

    samtools view -b ex1.bam > ex1_refresh.bam

    >>> handle = open("SamBam/ex1_refresh.bam", "rb")
    >>> for values in BgzfBlocks(handle):
    ...     print("Raw start %i, raw length %i; data start %i, data length %i" % values)
    Raw start 0, raw length 53; data start 0, data length 38
    Raw start 53, raw length 18195; data start 38, data length 65434
    Raw start 18248, raw length 18190; data start 65472, data length 65409
    Raw start 36438, raw length 18004; data start 130881, data length 65483
    Raw start 54442, raw length 17353; data start 196364, data length 65519
    Raw start 71795, raw length 17708; data start 261883, data length 65411
    Raw start 89503, raw length 17709; data start 327294, data length 65466
    Raw start 107212, raw length 17390; data start 392760, data length 63854
    Raw start 124602, raw length 28; data start 456614, data length 0
    >>> handle.close()

    The above example has no embedded SAM header (thus the first block
    is very small at just 38 bytes of decompressed data), while the next
    example does (a larger block of 103 bytes). Notice that the rest of
    the blocks show the same sizes (they contain the same read data):

    >>> handle = open("SamBam/ex1_header.bam", "rb")
    >>> for values in BgzfBlocks(handle):
    ...     print("Raw start %i, raw length %i; data start %i, data length %i" % values)
    Raw start 0, raw length 104; data start 0, data length 103
    Raw start 104, raw length 18195; data start 103, data length 65434
    Raw start 18299, raw length 18190; data start 65537, data length 65409
    Raw start 36489, raw length 18004; data start 130946, data length 65483
    Raw start 54493, raw length 17353; data start 196429, data length 65519
    Raw start 71846, raw length 17708; data start 261948, data length 65411
    Raw start 89554, raw length 17709; data start 327359, data length 65466
    Raw start 107263, raw length 17390; data start 392825, data length 63854
    Raw start 124653, raw length 28; data start 456679, data length 0
    >>> handle.close()

    """
    if isinstance(handle, BgzfReader):
        raise TypeError('Function BgzfBlocks expects a binary handle')
    data_start = 0
    while True:
        start_offset = handle.tell()
        try:
            block_length, data = _load_bgzf_block(handle)
        except StopIteration:
            break
        data_len = len(data)
        yield (start_offset, block_length, data_start, data_len)
        data_start += data_len
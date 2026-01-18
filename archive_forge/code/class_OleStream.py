from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
class OleStream(io.BytesIO):
    """
    OLE2 Stream

    Returns a read-only file object which can be used to read
    the contents of a OLE stream (instance of the BytesIO class).
    To open a stream, use the openstream method in the OleFileIO class.

    This function can be used with either ordinary streams,
    or ministreams, depending on the offset, sectorsize, and
    fat table arguments.

    Attributes:

        - size: actual size of data stream, after it was opened.
    """

    def __init__(self, fp, sect, size, offset, sectorsize, fat, filesize, olefileio):
        """
        Constructor for OleStream class.

        :param fp: file object, the OLE container or the MiniFAT stream
        :param sect: sector index of first sector in the stream
        :param size: total size of the stream
        :param offset: offset in bytes for the first FAT or MiniFAT sector
        :param sectorsize: size of one sector
        :param fat: array/list of sector indexes (FAT or MiniFAT)
        :param filesize: size of OLE file (for debugging)
        :param olefileio: OleFileIO object containing this stream
        :returns: a BytesIO instance containing the OLE stream
        """
        log.debug('OleStream.__init__:')
        log.debug('  sect=%d (%X), size=%d, offset=%d, sectorsize=%d, len(fat)=%d, fp=%s' % (sect, sect, size, offset, sectorsize, len(fat), repr(fp)))
        self.ole = olefileio
        if self.ole.fp.closed:
            raise OSError('Attempting to open a stream from a closed OLE File')
        unknown_size = False
        if size == UNKNOWN_SIZE:
            size = len(fat) * sectorsize
            unknown_size = True
            log.debug('  stream with UNKNOWN SIZE')
        nb_sectors = (size + (sectorsize - 1)) // sectorsize
        log.debug('nb_sectors = %d' % nb_sectors)
        if nb_sectors > len(fat):
            self.ole._raise_defect(DEFECT_INCORRECT, 'malformed OLE document, stream too large')
        data = []
        if size == 0 and sect != ENDOFCHAIN:
            log.debug('size == 0 and sect != ENDOFCHAIN:')
            self.ole._raise_defect(DEFECT_INCORRECT, 'incorrect OLE sector index for empty stream')
        for i in range(nb_sectors):
            log.debug('Reading stream sector[%d] = %Xh' % (i, sect))
            if sect == ENDOFCHAIN:
                if unknown_size:
                    log.debug('Reached ENDOFCHAIN sector for stream with unknown size')
                    break
                else:
                    log.debug('sect=ENDOFCHAIN before expected size')
                    self.ole._raise_defect(DEFECT_INCORRECT, 'incomplete OLE stream')
            if sect < 0 or sect >= len(fat):
                log.debug('sect=%d (%X) / len(fat)=%d' % (sect, sect, len(fat)))
                log.debug('i=%d / nb_sectors=%d' % (i, nb_sectors))
                self.ole._raise_defect(DEFECT_INCORRECT, 'incorrect OLE FAT, sector index out of range')
                break
            try:
                fp.seek(offset + sectorsize * sect)
            except Exception:
                log.debug('sect=%d, seek=%d, filesize=%d' % (sect, offset + sectorsize * sect, filesize))
                self.ole._raise_defect(DEFECT_INCORRECT, 'OLE sector index out of range')
                break
            sector_data = fp.read(sectorsize)
            if len(sector_data) != sectorsize and sect != len(fat) - 1:
                log.debug('sect=%d / len(fat)=%d, seek=%d / filesize=%d, len read=%d' % (sect, len(fat), offset + sectorsize * sect, filesize, len(sector_data)))
                log.debug('seek+len(read)=%d' % (offset + sectorsize * sect + len(sector_data)))
                self.ole._raise_defect(DEFECT_INCORRECT, 'incomplete OLE sector')
            data.append(sector_data)
            try:
                sect = fat[sect] & 4294967295
            except IndexError:
                self.ole._raise_defect(DEFECT_INCORRECT, 'incorrect OLE FAT, sector index out of range')
                break
        data = b''.join(data)
        if len(data) >= size:
            log.debug('Read data of length %d, truncated to stream size %d' % (len(data), size))
            data = data[:size]
            self.size = size
        elif unknown_size:
            log.debug('Read data of length %d, the stream size was unknown' % len(data))
            self.size = len(data)
        else:
            log.debug('Read data of length %d, less than expected stream size %d' % (len(data), size))
            self.size = len(data)
            self.ole._raise_defect(DEFECT_INCORRECT, 'OLE stream size is less than declared')
        io.BytesIO.__init__(self, data)
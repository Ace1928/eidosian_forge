import os
import json
import struct
import logging
import numpy as np
from ..core import Format
from ..v2 import imread
class LytroLfrFormat(LytroFormat):
    """This is the Lytro Illum LFR format.
    The lfr is a image and meta data container format as used by the
    Lytro Illum light field camera.
    The format will read the specified lfr file.
    This format does not support writing.

    Parameters for reading
    ----------------------
    meta_only : bool
        Whether to only read the metadata.
    include_thumbnail : bool
        Whether to include an image thumbnail in the metadata.
    """

    def _can_read(self, request):
        if request.extension in ('.lfr',):
            return True

    class Reader(Format.Reader):

        def _open(self, meta_only=False, include_thumbnail=True):
            self._file = self.request.get_file()
            self._data = None
            self._chunks = {}
            self.metadata = {}
            self._content = None
            self._meta_only = meta_only
            self._include_thumbnail = include_thumbnail
            self._find_header()
            self._find_chunks()
            self._find_meta()
            try:
                chunk_dict = self._content['frames'][0]['frame']
                if chunk_dict['metadataRef'] in self._chunks and chunk_dict['imageRef'] in self._chunks and (chunk_dict['privateMetadataRef'] in self._chunks):
                    if not self._meta_only:
                        data_pos, size = self._chunks[chunk_dict['imageRef']]
                        self._file.seek(data_pos, 0)
                        self.raw_image_data = self._file.read(size)
                    data_pos, size = self._chunks[chunk_dict['metadataRef']]
                    self._file.seek(data_pos, 0)
                    metadata = self._file.read(size)
                    self.metadata['metadata'] = json.loads(metadata.decode('ASCII'))
                    data_pos, size = self._chunks[chunk_dict['privateMetadataRef']]
                    self._file.seek(data_pos, 0)
                    serial_numbers = self._file.read(size)
                    self.serial_numbers = json.loads(serial_numbers.decode('ASCII'))
                    self.metadata['privateMetadata'] = self.serial_numbers
                if self._include_thumbnail:
                    chunk_dict = self._content['thumbnails'][0]
                    if chunk_dict['imageRef'] in self._chunks:
                        data_pos, size = self._chunks[chunk_dict['imageRef']]
                        self._file.seek(data_pos, 0)
                        thumbnail_data = self._file.read(size)
                        thumbnail_img = imread(thumbnail_data, format='jpeg')
                        thumbnail_height = chunk_dict['height']
                        thumbnail_width = chunk_dict['width']
                        self.metadata['thumbnail'] = {'image': thumbnail_img, 'height': thumbnail_height, 'width': thumbnail_width}
            except KeyError:
                raise RuntimeError('The specified file is not a valid LFR file.')

        def _close(self):
            del self._data

        def _get_length(self):
            return 1

        def _find_header(self):
            """
            Checks if file has correct header and skip it.
            """
            file_header = b'\x89LFP\r\n\x1a\n\x00\x00\x00\x01'
            header = self._file.read(HEADER_LENGTH)
            if header != file_header:
                raise RuntimeError('The LFR file header is invalid.')
            self._file.read(SIZE_LENGTH)

        def _find_chunks(self):
            """
            Gets start position and size of data chunks in file.
            """
            chunk_header = b'\x89LFC\r\n\x1a\n\x00\x00\x00\x00'
            for i in range(0, DATA_CHUNKS_ILLUM):
                data_pos, size, sha1 = self._get_chunk(chunk_header)
                self._chunks[sha1] = (data_pos, size)

        def _find_meta(self):
            """
            Gets a data chunk that contains information over content
            of other data chunks.
            """
            meta_header = b'\x89LFM\r\n\x1a\n\x00\x00\x00\x00'
            data_pos, size, sha1 = self._get_chunk(meta_header)
            self._file.seek(data_pos, 0)
            data = self._file.read(size)
            self._content = json.loads(data.decode('ASCII'))

        def _get_chunk(self, header):
            """
            Checks if chunk has correct header and skips it.
            Finds start position and length of next chunk and reads
            sha1-string that identifies the following data chunk.

            Parameters
            ----------
            header : bytes
                Byte string that identifies start of chunk.

            Returns
            -------
                data_pos : int
                    Start position of data chunk in file.
                size : int
                    Size of data chunk.
                sha1 : str
                    Sha1 value of chunk.
            """
            header_chunk = self._file.read(HEADER_LENGTH)
            if header_chunk != header:
                raise RuntimeError('The LFR chunk header is invalid.')
            data_pos = None
            sha1 = None
            size = struct.unpack('>i', self._file.read(SIZE_LENGTH))[0]
            if size > 0:
                sha1 = str(self._file.read(SHA1_LENGTH).decode('ASCII'))
                self._file.read(PADDING_LENGTH)
                data_pos = self._file.tell()
                self._file.seek(size, 1)
                ch = self._file.read(1)
                while ch == b'\x00':
                    ch = self._file.read(1)
                self._file.seek(-1, 1)
            return (data_pos, size, sha1)

        def _get_data(self, index):
            if index not in [0, None]:
                raise IndexError('Lytro lfr file contains only one dataset')
            if not self._meta_only:
                raw = np.frombuffer(self.raw_image_data, dtype=np.uint8).astype(np.uint16)
                im = LytroIllumRawFormat.rearrange_bits(raw)
            else:
                im = np.array([])
            return (im, self.metadata)

        def _get_meta_data(self, index):
            if index not in [0, None]:
                raise IndexError('Lytro meta data file contains only one dataset')
            return self.metadata
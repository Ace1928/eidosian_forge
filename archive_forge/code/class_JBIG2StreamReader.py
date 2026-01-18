import math
import os
from struct import pack, unpack, calcsize
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union, cast
class JBIG2StreamReader:
    """Read segments from a JBIG2 byte stream"""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream

    def get_segments(self) -> List[JBIG2Segment]:
        segments: List[JBIG2Segment] = []
        while not self.is_eof():
            segment: JBIG2Segment = {}
            for field_format, name in SEG_STRUCT:
                field_len = calcsize(field_format)
                field = self.stream.read(field_len)
                if len(field) < field_len:
                    segment['_error'] = True
                    break
                value = unpack_int(field_format, field)
                parser = getattr(self, 'parse_%s' % name, None)
                if callable(parser):
                    value = parser(segment, value, field)
                segment[name] = value
            if not segment.get('_error'):
                segments.append(segment)
        return segments

    def is_eof(self) -> bool:
        if self.stream.read(1) == b'':
            return True
        else:
            self.stream.seek(-1, os.SEEK_CUR)
            return False

    def parse_flags(self, segment: JBIG2Segment, flags: int, field: bytes) -> JBIG2SegmentFlags:
        return {'deferred': check_flag(HEADER_FLAG_DEFERRED, flags), 'page_assoc_long': check_flag(HEADER_FLAG_PAGE_ASSOC_LONG, flags), 'type': masked_value(SEG_TYPE_MASK, flags)}

    def parse_retention_flags(self, segment: JBIG2Segment, flags: int, field: bytes) -> JBIG2RetentionFlags:
        ref_count = masked_value(REF_COUNT_SHORT_MASK, flags)
        retain_segments = []
        ref_segments = []
        if ref_count < REF_COUNT_LONG:
            for bit_pos in range(5):
                retain_segments.append(bit_set(bit_pos, flags))
        else:
            field += self.stream.read(3)
            ref_count = unpack_int('>L', field)
            ref_count = masked_value(REF_COUNT_LONG_MASK, ref_count)
            ret_bytes_count = int(math.ceil((ref_count + 1) / 8))
            for ret_byte_index in range(ret_bytes_count):
                ret_byte = unpack_int('>B', self.stream.read(1))
                for bit_pos in range(7):
                    retain_segments.append(bit_set(bit_pos, ret_byte))
        seg_num = segment['number']
        assert isinstance(seg_num, int)
        if seg_num <= 256:
            ref_format = '>B'
        elif seg_num <= 65536:
            ref_format = '>I'
        else:
            ref_format = '>L'
        ref_size = calcsize(ref_format)
        for ref_index in range(ref_count):
            ref_data = self.stream.read(ref_size)
            ref = unpack_int(ref_format, ref_data)
            ref_segments.append(ref)
        return {'ref_count': ref_count, 'retain_segments': retain_segments, 'ref_segments': ref_segments}

    def parse_page_assoc(self, segment: JBIG2Segment, page: int, field: bytes) -> int:
        if cast(JBIG2SegmentFlags, segment['flags'])['page_assoc_long']:
            field += self.stream.read(3)
            page = unpack_int('>L', field)
        return page

    def parse_data_length(self, segment: JBIG2Segment, length: int, field: bytes) -> int:
        if length:
            if cast(JBIG2SegmentFlags, segment['flags'])['type'] == SEG_TYPE_IMMEDIATE_GEN_REGION and length == DATA_LEN_UNKNOWN:
                raise NotImplementedError('Working with unknown segment length is not implemented yet')
            else:
                segment['raw_data'] = self.stream.read(length)
        return length
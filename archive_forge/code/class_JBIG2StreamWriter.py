import math
import os
from struct import pack, unpack, calcsize
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union, cast
class JBIG2StreamWriter:
    """Write JBIG2 segments to a file in JBIG2 format"""
    EMPTY_RETENTION_FLAGS: JBIG2RetentionFlags = {'ref_count': 0, 'ref_segments': cast(List[int], []), 'retain_segments': cast(List[bool], [])}

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream

    def write_segments(self, segments: Iterable[JBIG2Segment], fix_last_page: bool=True) -> int:
        data_len = 0
        current_page: Optional[int] = None
        seg_num: Optional[int] = None
        for segment in segments:
            data = self.encode_segment(segment)
            self.stream.write(data)
            data_len += len(data)
            seg_num = cast(Optional[int], segment['number'])
            if fix_last_page:
                seg_page = cast(int, segment.get('page_assoc'))
                if cast(JBIG2SegmentFlags, segment['flags'])['type'] == SEG_TYPE_END_OF_PAGE:
                    current_page = None
                elif seg_page:
                    current_page = seg_page
        if fix_last_page and current_page and (seg_num is not None):
            segment = self.get_eop_segment(seg_num + 1, current_page)
            data = self.encode_segment(segment)
            self.stream.write(data)
            data_len += len(data)
        return data_len

    def write_file(self, segments: Iterable[JBIG2Segment], fix_last_page: bool=True) -> int:
        header = FILE_HEADER_ID
        header_flags = FILE_HEAD_FLAG_SEQUENTIAL
        header += pack('>B', header_flags)
        number_of_pages = pack('>L', 1)
        header += number_of_pages
        self.stream.write(header)
        data_len = len(header)
        data_len += self.write_segments(segments, fix_last_page)
        seg_num = 0
        for segment in segments:
            seg_num = cast(int, segment['number'])
        if fix_last_page:
            seg_num_offset = 2
        else:
            seg_num_offset = 1
        eof_segment = self.get_eof_segment(seg_num + seg_num_offset)
        data = self.encode_segment(eof_segment)
        self.stream.write(data)
        data_len += len(data)
        return data_len

    def encode_segment(self, segment: JBIG2Segment) -> bytes:
        data = b''
        for field_format, name in SEG_STRUCT:
            value = segment.get(name)
            encoder = getattr(self, 'encode_%s' % name, None)
            if callable(encoder):
                field = encoder(value, segment)
            else:
                field = pack(field_format, value)
            data += field
        return data

    def encode_flags(self, value: JBIG2SegmentFlags, segment: JBIG2Segment) -> bytes:
        flags = 0
        if value.get('deferred'):
            flags |= HEADER_FLAG_DEFERRED
        if 'page_assoc_long' in value:
            flags |= HEADER_FLAG_PAGE_ASSOC_LONG if value['page_assoc_long'] else flags
        else:
            flags |= HEADER_FLAG_PAGE_ASSOC_LONG if cast(int, segment.get('page', 0)) > 255 else flags
        flags |= mask_value(SEG_TYPE_MASK, value['type'])
        return pack('>B', flags)

    def encode_retention_flags(self, value: JBIG2RetentionFlags, segment: JBIG2Segment) -> bytes:
        flags = []
        flags_format = '>B'
        ref_count = value['ref_count']
        assert isinstance(ref_count, int)
        retain_segments = cast(List[bool], value.get('retain_segments', []))
        if ref_count <= 4:
            flags_byte = mask_value(REF_COUNT_SHORT_MASK, ref_count)
            for ref_index, ref_retain in enumerate(retain_segments):
                if ref_retain:
                    flags_byte |= 1 << ref_index
            flags.append(flags_byte)
        else:
            bytes_count = math.ceil((ref_count + 1) / 8)
            flags_format = '>L' + 'B' * bytes_count
            flags_dword = mask_value(REF_COUNT_SHORT_MASK, REF_COUNT_LONG) << 24
            flags.append(flags_dword)
            for byte_index in range(bytes_count):
                ret_byte = 0
                ret_part = retain_segments[byte_index * 8:byte_index * 8 + 8]
                for bit_pos, ret_seg in enumerate(ret_part):
                    ret_byte |= 1 << bit_pos if ret_seg else ret_byte
                flags.append(ret_byte)
        ref_segments = cast(List[int], value.get('ref_segments', []))
        seg_num = cast(int, segment['number'])
        if seg_num <= 256:
            ref_format = 'B'
        elif seg_num <= 65536:
            ref_format = 'I'
        else:
            ref_format = 'L'
        for ref in ref_segments:
            flags_format += ref_format
            flags.append(ref)
        return pack(flags_format, *flags)

    def encode_data_length(self, value: int, segment: JBIG2Segment) -> bytes:
        data = pack('>L', value)
        data += cast(bytes, segment['raw_data'])
        return data

    def get_eop_segment(self, seg_number: int, page_number: int) -> JBIG2Segment:
        return {'data_length': 0, 'flags': {'deferred': False, 'type': SEG_TYPE_END_OF_PAGE}, 'number': seg_number, 'page_assoc': page_number, 'raw_data': b'', 'retention_flags': JBIG2StreamWriter.EMPTY_RETENTION_FLAGS}

    def get_eof_segment(self, seg_number: int) -> JBIG2Segment:
        return {'data_length': 0, 'flags': {'deferred': False, 'type': SEG_TYPE_END_OF_FILE}, 'number': seg_number, 'page_assoc': 0, 'raw_data': b'', 'retention_flags': JBIG2StreamWriter.EMPTY_RETENTION_FLAGS}
import os
from ..construct.macros import UBInt32, UBInt64, ULInt32, ULInt64, Array
from ..common.exceptions import DWARFError
from ..common.utils import preserve_stream_pos, struct_parse
def _iter_CUs_in_section(stream, structs, parser):
    """Iterates through the list of CU sections in loclists or rangelists. Almost identical structures there.

    get_parser is a lambda that takes structs, returns the parser
    """
    stream.seek(0, os.SEEK_END)
    endpos = stream.tell()
    stream.seek(0, os.SEEK_SET)
    offset = 0
    while offset < endpos:
        header = struct_parse(parser, stream, offset)
        if header.offset_count > 0:
            offset_parser = structs.Dwarf_uint64 if header.is64 else structs.Dwarf_uint32
            header['offsets'] = struct_parse(Array(header.offset_count, offset_parser('')), stream)
        else:
            header['offsets'] = False
        yield header
        offset = header.offset_after_length + header.unit_length
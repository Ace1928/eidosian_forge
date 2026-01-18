import os
from ..construct.macros import UBInt32, UBInt64, ULInt32, ULInt64, Array
from ..common.exceptions import DWARFError
from ..common.utils import preserve_stream_pos, struct_parse
Iterates through the list of CU sections in loclists or rangelists. Almost identical structures there.

    get_parser is a lambda that takes structs, returns the parser
    
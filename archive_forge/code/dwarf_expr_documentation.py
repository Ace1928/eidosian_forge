from collections import namedtuple
from io import BytesIO
from ..common.utils import struct_parse, bytelist2string, read_blob
from ..common.exceptions import DWARFError
 Parses expr (a list of integers) into a list of DWARFExprOp.

        The list can potentially be nested.
        
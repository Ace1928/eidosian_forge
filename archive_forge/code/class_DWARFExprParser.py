from collections import namedtuple
from io import BytesIO
from ..common.utils import struct_parse, bytelist2string, read_blob
from ..common.exceptions import DWARFError
class DWARFExprParser(object):
    """DWARF expression parser.

    When initialized, requires structs to cache a dispatch table. After that,
    parse_expr can be called repeatedly - it's stateless.
    """

    def __init__(self, structs):
        self._dispatch_table = _init_dispatch_table(structs)

    def parse_expr(self, expr):
        """ Parses expr (a list of integers) into a list of DWARFExprOp.

        The list can potentially be nested.
        """
        stream = BytesIO(bytelist2string(expr))
        parsed = []
        while True:
            offset = stream.tell()
            byte = stream.read(1)
            if len(byte) == 0:
                break
            op = ord(byte)
            op_name = DW_OP_opcode2name.get(op, 'OP:0x%x' % op)
            arg_parser = self._dispatch_table[op]
            args = arg_parser(stream)
            parsed.append(DWARFExprOp(op=op, op_name=op_name, args=args, offset=offset))
        return parsed
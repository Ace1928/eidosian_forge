import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
def _process_next(self, i_line):
    instruction = self.instructions.pop(0)
    handler_class = _op_name_to_handler.get(instruction.opname)
    if handler_class is not None:
        s = handler_class(i_line, instruction, self.stack, self.writer, self)
        if DEBUG:
            print(s)
    elif DEBUG:
        print('UNHANDLED', instruction)
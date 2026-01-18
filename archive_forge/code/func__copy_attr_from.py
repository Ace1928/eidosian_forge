import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import UNSET, Label, SetLineno, Instr
from _pydevd_frame_eval.vendored.bytecode.flags import infer_flags
def _copy_attr_from(self, bytecode):
    super()._copy_attr_from(bytecode)
    if isinstance(bytecode, Bytecode):
        self.argnames = bytecode.argnames
import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.concrete import ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.flags import CompilerFlags
from _pydevd_frame_eval.vendored.bytecode.instr import Label, SetLineno, Instr
def _add_block(self, block):
    block_index = len(self._blocks)
    self._blocks.append(block)
    self._block_index[id(block)] = block_index
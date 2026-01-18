import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.concrete import ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.flags import CompilerFlags
from _pydevd_frame_eval.vendored.bytecode.instr import Label, SetLineno, Instr
class BasicBlock(_bytecode._InstrList):

    def __init__(self, instructions=None):
        self.next_block = None
        if instructions:
            super().__init__(instructions)

    def __iter__(self):
        index = 0
        while index < len(self):
            instr = self[index]
            index += 1
            if not isinstance(instr, (SetLineno, Instr)):
                raise ValueError('BasicBlock must only contain SetLineno and Instr objects, but %s was found' % instr.__class__.__name__)
            if isinstance(instr, Instr) and instr.has_jump():
                if index < len(self):
                    raise ValueError('Only the last instruction of a basic block can be a jump')
                if not isinstance(instr.arg, BasicBlock):
                    raise ValueError('Jump target must a BasicBlock, got %s', type(instr.arg).__name__)
            yield instr

    def __getitem__(self, index):
        value = super().__getitem__(index)
        if isinstance(index, slice):
            value = type(self)(value)
            value.next_block = self.next_block
        return value

    def copy(self):
        new = type(self)(super().copy())
        new.next_block = self.next_block
        return new

    def legalize(self, first_lineno):
        """Check that all the element of the list are valid and remove SetLineno."""
        lineno_pos = []
        set_lineno = None
        current_lineno = first_lineno
        for pos, instr in enumerate(self):
            if isinstance(instr, SetLineno):
                set_lineno = current_lineno = instr.lineno
                lineno_pos.append(pos)
                continue
            if set_lineno is not None:
                instr.lineno = set_lineno
            elif instr.lineno is None:
                instr.lineno = current_lineno
            else:
                current_lineno = instr.lineno
        for i in reversed(lineno_pos):
            del self[i]
        return current_lineno

    def get_jump(self):
        if not self:
            return None
        last_instr = self[-1]
        if not (isinstance(last_instr, Instr) and last_instr.has_jump()):
            return None
        target_block = last_instr.arg
        assert isinstance(target_block, BasicBlock)
        return target_block
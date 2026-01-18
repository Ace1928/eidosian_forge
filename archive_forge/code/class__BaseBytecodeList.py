import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import UNSET, Label, SetLineno, Instr
from _pydevd_frame_eval.vendored.bytecode.flags import infer_flags
class _BaseBytecodeList(BaseBytecode, list):
    """List subclass providing type stable slicing and copying."""

    def __getitem__(self, index):
        value = super().__getitem__(index)
        if isinstance(index, slice):
            value = type(self)(value)
            value._copy_attr_from(self)
        return value

    def copy(self):
        new = type(self)(super().copy())
        new._copy_attr_from(self)
        return new

    def legalize(self):
        """Check that all the element of the list are valid and remove SetLineno."""
        lineno_pos = []
        set_lineno = None
        current_lineno = self.first_lineno
        for pos, instr in enumerate(self):
            if isinstance(instr, SetLineno):
                set_lineno = instr.lineno
                lineno_pos.append(pos)
                continue
            if not isinstance(instr, Instr):
                continue
            if set_lineno is not None:
                instr.lineno = set_lineno
            elif instr.lineno is None:
                instr.lineno = current_lineno
            else:
                current_lineno = instr.lineno
        for i in reversed(lineno_pos):
            del self[i]

    def __iter__(self):
        instructions = super().__iter__()
        for instr in instructions:
            self._check_instr(instr)
            yield instr

    def _check_instr(self, instr):
        raise NotImplementedError()
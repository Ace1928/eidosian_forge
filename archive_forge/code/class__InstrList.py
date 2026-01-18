import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import UNSET, Label, SetLineno, Instr
from _pydevd_frame_eval.vendored.bytecode.flags import infer_flags
class _InstrList(list):

    def _flat(self):
        instructions = []
        labels = {}
        jumps = []
        offset = 0
        for index, instr in enumerate(self):
            if isinstance(instr, Label):
                instructions.append('label_instr%s' % index)
                labels[instr] = offset
            else:
                if isinstance(instr, Instr) and isinstance(instr.arg, Label):
                    target_label = instr.arg
                    instr = _bytecode.ConcreteInstr(instr.name, 0, lineno=instr.lineno)
                    jumps.append((target_label, instr))
                instructions.append(instr)
                offset += 1
        for target_label, instr in jumps:
            instr.arg = labels[target_label]
        return instructions

    def __eq__(self, other):
        if not isinstance(other, _InstrList):
            other = _InstrList(other)
        return self._flat() == other._flat()
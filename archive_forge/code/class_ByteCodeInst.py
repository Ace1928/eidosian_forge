from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
class ByteCodeInst(object):
    """
    Attributes
    ----------
    - offset:
        byte offset of opcode
    - opcode:
        opcode integer value
    - arg:
        instruction arg
    - lineno:
        -1 means unknown
    """
    __slots__ = ('offset', 'next', 'opcode', 'opname', 'arg', 'lineno')

    def __init__(self, offset, opcode, arg, nextoffset):
        self.offset = offset
        self.next = nextoffset
        self.opcode = opcode
        self.opname = dis.opname[opcode]
        self.arg = arg
        self.lineno = -1

    @property
    def is_jump(self):
        return self.opcode in JUMP_OPS

    @property
    def is_terminator(self):
        return self.opcode in TERM_OPS

    def get_jump_target(self):
        assert self.is_jump
        if PYVERSION in ((3, 12),):
            if self.opcode in (dis.opmap[k] for k in ['JUMP_BACKWARD']):
                return self.offset - (self.arg - 1) * 2
        elif PYVERSION in ((3, 11),):
            if self.opcode in (dis.opmap[k] for k in ('JUMP_BACKWARD', 'POP_JUMP_BACKWARD_IF_TRUE', 'POP_JUMP_BACKWARD_IF_FALSE', 'POP_JUMP_BACKWARD_IF_NONE', 'POP_JUMP_BACKWARD_IF_NOT_NONE')):
                return self.offset - (self.arg - 1) * 2
        elif PYVERSION in ((3, 9), (3, 10)):
            pass
        else:
            raise NotImplementedError(PYVERSION)
        if PYVERSION in ((3, 10), (3, 11), (3, 12)):
            if self.opcode in JREL_OPS:
                return self.next + self.arg * 2
            else:
                assert self.opcode in JABS_OPS
                return self.arg * 2 - 2
        elif PYVERSION in ((3, 9),):
            if self.opcode in JREL_OPS:
                return self.next + self.arg
            else:
                assert self.opcode in JABS_OPS
                return self.arg
        else:
            raise NotImplementedError(PYVERSION)

    def __repr__(self):
        return '%s(arg=%s, lineno=%d)' % (self.opname, self.arg, self.lineno)

    @property
    def block_effect(self):
        """Effect of the block stack
        Returns +1 (push), 0 (none) or -1 (pop)
        """
        if self.opname.startswith('SETUP_'):
            return 1
        elif self.opname == 'POP_BLOCK':
            return -1
        else:
            return 0
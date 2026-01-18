from . import DefaultTable
from . import ttProgram

        >>> fpgm = table__f_p_g_m()
        >>> bool(fpgm)
        False
        >>> p = ttProgram.Program()
        >>> fpgm.program = p
        >>> bool(fpgm)
        False
        >>> bc = bytearray([0])
        >>> p.fromBytecode(bc)
        >>> bool(fpgm)
        True
        >>> p.bytecode.pop()
        0
        >>> bool(fpgm)
        False
        
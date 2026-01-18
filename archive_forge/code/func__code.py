import _sre
from . import _parser
from ._constants import *
from ._casefix import _EXTRA_CASES
def _code(p, flags):
    flags = p.state.flags | flags
    code = []
    _compile_info(code, p, flags)
    _compile(code, p.data, flags)
    code.append(SUCCESS)
    return code
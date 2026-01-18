import operator
import warnings
from llvmlite import ir
from numba.core import types, cgutils
from numba.core import typing
from numba.core.registry import cpu_target
from numba.core.typeconv import Conversion
from numba.core.extending import intrinsic
from numba.core.errors import TypingError, NumbaTypeSafetyWarning
def _sentry_safe_cast(fromty, toty):
    """Check and raise TypingError if *fromty* cannot be safely cast to *toty*
    """
    tyctxt = cpu_target.typing_context
    fromty, toty = map(types.unliteral, (fromty, toty))
    by = tyctxt.can_convert(fromty, toty)

    def warn():
        m = 'unsafe cast from {} to {}. Precision may be lost.'
        warnings.warn(m.format(fromty, toty), category=NumbaTypeSafetyWarning)
    isint = lambda x: isinstance(x, types.Integer)
    isflt = lambda x: isinstance(x, types.Float)
    iscmplx = lambda x: isinstance(x, types.Complex)
    isdict = lambda x: isinstance(x, types.DictType)
    if by is None or by > Conversion.safe:
        if isint(fromty) and isint(toty):
            warn()
        elif isint(fromty) and isflt(toty):
            warn()
        elif isflt(fromty) and isflt(toty):
            warn()
        elif iscmplx(fromty) and iscmplx(toty):
            warn()
        elif isdict(fromty) and isdict(toty):
            pass
        elif not isinstance(toty, types.Number):
            warn()
        else:
            m = 'cannot safely cast {} to {}. Please cast explicitly.'
            raise TypingError(m.format(fromty, toty))
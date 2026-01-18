from .z3 import *
from .z3core import *
from .z3printer import *
def MkRoots(p, ctx=None):
    ctx = z3.get_ctx(ctx)
    num = len(p)
    _tmp = []
    _as = (RCFNumObj * num)()
    _rs = (RCFNumObj * num)()
    for i in range(num):
        _a = _to_rcfnum(p[i], ctx)
        _tmp.append(_a)
        _as[i] = _a.num
    nr = Z3_rcf_mk_roots(ctx.ref(), num, _as, _rs)
    r = []
    for i in range(nr):
        r.append(RCFNum(_rs[i], ctx))
    return r
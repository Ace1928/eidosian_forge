import linecache
import re
from inspect import (getblock, getfile, getmodule, getsourcefile, indentsize,
from tokenize import TokenError
from ._dill import IS_IPYTHON
def _matchlambda(func, line):
    """check if lambda object 'func' matches raw line of code 'line'"""
    from .detect import code as getcode
    from .detect import freevars, globalvars, varnames
    dummy = lambda: '__this_is_a_big_dummy_function__'
    lhs, rhs = line.split('lambda ', 1)[-1].split(':', 1)
    try:
        _ = eval('lambda %s : %s' % (lhs, rhs), globals(), locals())
    except Exception:
        _ = dummy
    _, code = (getcode(_).co_code, getcode(func).co_code)
    _f = [line.count(i) for i in freevars(func).keys()]
    if not _f:
        if _ == code:
            return True
        return False
    if not all(_f):
        return False
    _f = varnames(func)
    _f = [line.count(i) for i in _f[0] + _f[1]]
    if _f and (not all(_f)):
        return False
    _f = [line.count(i) for i in globalvars(func).keys()]
    if _f and (not all(_f)):
        return False
    if line.count('lambda ') > 1 and lhs in freevars(func).keys():
        _lhs, _rhs = rhs.split('lambda ', 1)[-1].split(':', 1)
        try:
            _f = eval('lambda %s : %s' % (_lhs, _rhs), globals(), locals())
        except Exception:
            _f = dummy
        _, code = (getcode(_f).co_code, getcode(func).co_code)
        if len(_) != len(code):
            return False
        _ = set(((i, j) for i, j in zip(_, code) if i != j))
        if len(_) != 1:
            return False
        return True
    if not indentsize(line):
        return False
    _ = _.split(_[0])
    _f = code.split(code[0])
    _ = dict((re.match('([\\W\\D\\S])(.*)', _[i]).groups() for i in range(1, len(_))))
    _f = dict((re.match('([\\W\\D\\S])(.*)', _f[i]).groups() for i in range(1, len(_f))))
    if _.keys() == _f.keys() and sorted(_.values()) == sorted(_f.values()):
        return True
    return False
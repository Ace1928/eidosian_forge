from functools import partial
def _setOpt(name, value, conv=None, chk=None):
    """set a module level value from environ/default"""
    from os import environ
    ename = 'RL_' + name
    if ename in environ:
        value = environ[ename]
    if conv:
        value = conv(value)
    chk = _rlChecks.get(name, None)
    if chk:
        chk(name, value)
    globals()[name] = value
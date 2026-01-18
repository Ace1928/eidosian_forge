import functools
import inspect
import sys
from pyomo.common import DeveloperError
def _disable_method(fcn, msg=None, exception=RuntimeError):
    _name = fcn.__name__
    if msg is None:
        msg = "access '%s' on" % (_name,)
    sig = inspect.signature(fcn)
    params = list(sig.parameters.values())
    for i, param in enumerate(params):
        if param.default is inspect.Parameter.empty:
            continue
        params[i] = param.replace(default=None)
    sig = sig.replace(parameters=params)
    args = str(sig)
    assert args == '(self)' or args.startswith('(self,')
    _env = {'_msg': msg, '_name': _name}
    _funcdef = 'def %s%s:\n        raise %s("%s" %% (_msg, type(self).__name__,\n            self.name, _name, self.name))\n' % (_name, args, exception.__name__, _disabled_error)
    exec(_funcdef, _env)
    return functools.wraps(fcn)(_env[_name])
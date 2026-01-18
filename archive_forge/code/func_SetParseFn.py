from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
def SetParseFn(fn, *arguments):
    """Sets the fn for Fire to use to parse args when calling the decorated fn.

  Args:
    fn: The function to be used for parsing arguments.
    *arguments: The arguments for which to use the parse fn. If none are listed,
      then this will set the default parse function.
  Returns:
    The decorated function, which now has metadata telling Fire how to perform.
  """

    def _Decorator(func):
        parse_fns = GetParseFns(func)
        if not arguments:
            parse_fns['default'] = fn
        else:
            for argument in arguments:
                parse_fns['named'][argument] = fn
        _SetMetadata(func, FIRE_PARSE_FNS, parse_fns)
        return func
    return _Decorator
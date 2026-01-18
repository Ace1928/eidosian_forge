import calendar
from typing import Any, Optional, Tuple
class IntegerRange(Integer):

    def __init__(self, name, min, max, allowNone=1, default=None, shortDesc=None, longDesc=None, hints=None):
        self.min = min
        self.max = max
        Integer.__init__(self, name, allowNone=allowNone, default=default, shortDesc=shortDesc, longDesc=longDesc, hints=hints)

    def coerce(self, val):
        result = Integer.coerce(self, val)
        if self.allowNone and result == None:
            return result
        if result < self.min:
            raise InputError('Value {} is too small, it should be at least {}'.format(result, self.min))
        if result > self.max:
            raise InputError('Value {} is too large, it should be at most {}'.format(result, self.max))
        return result
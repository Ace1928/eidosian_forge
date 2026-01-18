from math import hypot, sqrt
from functools import wraps
from itertools import repeat
class bound(object):
    """Decorator for crossover and mutation functions, it changes the
    individuals after the modification is done to bring it back in the allowed
    *bounds*. The *bounds* are functions taking individual and returning
    whether of not the variable is allowed. You can provide one or multiple such
    functions. In the former case, the function is used on all dimensions and
    in the latter case, the number of functions must be greater or equal to
    the number of dimension of the individuals.

    The *type* determines how the attributes are brought back into the valid
    range

    This decorator adds a :func:`bound` method to the decorated function.
    """

    def _clip(self, individual):
        return individual

    def _wrap(self, individual):
        return individual

    def _mirror(self, individual):
        return individual

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kargs):
            individuals = func(*args, **kargs)
            return self.bound(individuals)
        wrapper.bound = self.bound
        return wrapper

    def __init__(self, bounds, type):
        try:
            self.bounds = tuple(bounds)
        except TypeError:
            self.bounds = repeat(bounds)
        if type == 'mirror':
            self.bound = self._mirror
        elif type == 'wrap':
            self.bound = self._wrap
        elif type == 'clip':
            self.bound = self._clip
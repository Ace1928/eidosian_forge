from sympy.core.basic import Basic
from sympy.core.symbol import (Symbol, symbols)
from sympy.utilities.lambdify import lambdify
from .util import interpolate, rinterpolate, create_bounds, update_bounds
from sympy.utilities.iterables import sift
def _test_color_function(self):
    if not callable(self.f):
        raise ValueError('Color function is not callable.')
    try:
        result = self.f(0, 0, 0, 0, 0)
        if len(result) != 3:
            raise ValueError('length should be equal to 3')
    except TypeError:
        raise ValueError("Color function needs to accept x,y,z,u,v, as arguments even if it doesn't use all of them.")
    except AssertionError:
        raise ValueError('Color function needs to return 3-tuple r,g,b.')
    except Exception:
        pass
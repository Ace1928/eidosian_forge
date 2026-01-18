from functools import update_wrapper
import numpy as np
Decorator for being like an array without being an array.

    Poke decorators onto cls so that getting an attribute
    really gets that attribute from the wrapped ndarray.

    Exceptions are made for in-place methods like +=, *=, etc.
    These must return self since otherwise myobj += 1 would
    magically turn into an ndarray.
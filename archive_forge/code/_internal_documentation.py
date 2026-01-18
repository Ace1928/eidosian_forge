from functools import wraps
from inspect import signature

    Decorator to automatically replace xp with the corresponding array module.

    Use like

    import numpy as np

    @get_xp(np)
    def func(x, /, xp, kwarg=None):
        return xp.func(x, kwarg=kwarg)

    Note that xp must be a keyword argument and come after all non-keyword
    arguments.

    
import sys, os, subprocess
from .error import PkgConfigError
def _macro(x):
    x = x[2:]
    if '=' in x:
        return tuple(x.split('=', 1))
    else:
        return (x, None)
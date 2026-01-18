import common_z3 as CM_Z3
import ctypes
from .z3 import *
def Iff(f):
    return And(Implies(f[0], f[1]), Implies(f[1], f[0]))
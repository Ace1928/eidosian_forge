import math
import sympy
import torch
class PythonReferenceAnalysis(ReferenceAnalysis):

    @staticmethod
    def constant(c, dtype):
        if dtype is torch.int64:
            return int(c)
        elif dtype is torch.double:
            return float(c)
        elif dtype is torch.bool:
            return bool(c)
        else:
            raise AssertionError(f'unrecognized dtype {dtype}')

    @staticmethod
    def not_(a):
        return torch.sym_not(a)

    @staticmethod
    def floordiv(a, b):
        return a // b

    @staticmethod
    def truncdiv(a, b):
        return a / b

    @staticmethod
    def exp(x):
        raise AssertionError('exp is not valid shape sympy expr')

    @staticmethod
    def log(x):
        raise AssertionError('log is not valid shape sympy expr')

    @staticmethod
    def sqrt(x):
        return torch.sym_sqrt(x)

    @staticmethod
    def minimum(a, b):
        return torch.sym_min(a, b)

    @staticmethod
    def maximum(a, b):
        return torch.sym_max(a, b)

    @staticmethod
    def floor(x):
        return math.floor(x)

    @staticmethod
    def ceil(x):
        return math.ceil(x)
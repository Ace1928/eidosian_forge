import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def _patch_lang_math(lang, builder):
    math = lang.math
    mapping = {'abs': 'abs', 'acos': 'arccos', 'asin': 'arcsin', 'exp2': 'exp2', 'log2': 'log2', 'max': 'maximum'}

    def make_numpy(name):

        def impl(*args, **kwargs):
            ret_type = args[0].type
            ret_dtype = args[0].dtype
            args = [arg.handle.data for arg in args]
            kwargs = {k: v.handle.data for k, v in kwargs.items()}
            ret = getattr(np, mapping[name])(*args, **kwargs)
            ret = tl.core.tensor(TensorHandle(ret, ret_dtype), ret_type)
            return ret
        return impl

    def make_fallback(name):

        def fallback(*args, **kwargs):
            raise NotImplementedError(f'\n{name} not supported in interpreter mode: no known numpy implementation.\nIf you think that {name} in fact does have a numpy implementation, please add it\nto the mapping in python/triton/interpreter/new_interpreter.py:_patch_lang_math.\n')
        return fallback
    for name, member in inspect.getmembers(math):
        if name in mapping:
            setattr(math, name, make_numpy(name))
        else:
            setattr(math, name, make_fallback(name))
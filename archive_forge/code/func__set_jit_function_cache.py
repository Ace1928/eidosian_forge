import os
import weakref
import torch
def _set_jit_function_cache(key, value):
    assert isinstance(value, torch.jit.ScriptFunction)
    _jit_caching_layer[key] = value.qualified_name
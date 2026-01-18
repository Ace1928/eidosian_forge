import sys
from collections import namedtuple
def _handler(**kwargs):
    module, function_name = full_name.rsplit('.', 1)
    module = import_module(module)
    kwargs.update(parent_kwargs)
    return getattr(module, function_name)(**kwargs)
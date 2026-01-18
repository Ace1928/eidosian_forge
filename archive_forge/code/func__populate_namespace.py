from typing import Dict
from .core import ConstExpression
def _populate_namespace():
    globals_ = globals()
    for name, doc in CONST_LISTING.items():
        py_name = NAME_MAP.get(name, name)
        globals_[py_name] = ConstExpression(name, doc)
        yield py_name
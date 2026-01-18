import inspect, re
import types
from typing import Optional, Callable
from lark import Transformer, v_args
class WithMeta:
    """Abstract class

    Subclasses will be instantiated with the Meta instance of the tree. (see ``v_args`` for more detail)
    """
    pass
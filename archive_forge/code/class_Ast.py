import inspect, re
import types
from typing import Optional, Callable
from lark import Transformer, v_args
class Ast:
    """Abstract class

    Subclasses will be collected by `create_transformer()`
    """
    pass
import inspect, re
import types
from typing import Optional, Callable
from lark import Transformer, v_args
Collects `Ast` subclasses from the given module, and creates a Lark transformer that builds the AST.

    For each class, we create a corresponding rule in the transformer, with a matching name.
    CamelCase names will be converted into snake_case. Example: "CodeBlock" -> "code_block".

    Classes starting with an underscore (`_`) will be skipped.

    Parameters:
        ast_module: A Python module containing all the subclasses of ``ast_utils.Ast``
        transformer (Optional[Transformer]): An initial transformer. Its attributes may be overwritten.
        decorator_factory (Callable): An optional callable accepting two booleans, inline, and meta,
            and returning a decorator for the methods of ``transformer``. (default: ``v_args``).
    
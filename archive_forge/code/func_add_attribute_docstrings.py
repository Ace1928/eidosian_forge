import ast
import inspect
import sys
import textwrap
import typing as T
from types import ModuleType
from .common import Docstring, DocstringParam
def add_attribute_docstrings(obj: T.Union[type, ModuleType], docstring: Docstring) -> None:
    """Add attribute docstrings found in the object's source code.

    :param obj: object from which to parse attribute docstrings
    :param docstring: Docstring object where found attributes are added
    :returns: list with names of added attributes
    """
    params = set((p.arg_name for p in docstring.params))
    for arg_name, (description, type_name, default) in AttributeDocstrings().get_attr_docs(obj).items():
        if arg_name not in params:
            param = DocstringParam(args=['attribute', arg_name], description=description, arg_name=arg_name, type_name=type_name, is_optional=default is not None, default=default)
            docstring.meta.append(param)
import ast  # Importing the abstract syntax tree module to manipulate and analyze Python abstract syntax grammatically. Documentation: https://docs.python.org/3/library/ast.html
import json  # Importing the JSON module to encode and decode JSON data. Documentation: https://docs.python.org/3/library/json.html
import tkinter as tk  # Importing the tkinter module for creating graphical user interfaces. Documentation: https://docs.python.org/3/library/tkinter.html
from tkinter import (
import os  # Importing the os module to interact with the operating system. Documentation: https://docs.python.org/3/library/os.html
import logging  # Importing the logging module to enable logging capabilities. Documentation: https://docs.python.org/3/library/logging.html
from typing import (
import docstring_parser  # Importing the docstring_parser module to parse Python docstrings. Documentation: https://pypi.org/project/docstring-parser/
from concurrent.futures import (
def _infer_arg_type(self, assign: ast.arg) -> str:
    """
        Enhanced to handle complex types using the _infer_complex_type method, including nested generics and Python 3.10+ features like TypeGuard.

        Parameters:
            assign (ast.arg): The argument node whose type is to be inferred.

        Returns:
            str: A string representation of the inferred type for the argument.

        This method logs the inferred type for each argument and uses 'Any' as a default type if no annotation is present.
        """
    if assign.annotation:
        inferred_type: str = self._infer_complex_type(assign.annotation)
        logging.debug(f"Inferred type for argument '{assign.arg}': {inferred_type}")
        return inferred_type
    return 'Any'
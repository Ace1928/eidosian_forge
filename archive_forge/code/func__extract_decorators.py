import ast  # Importing the abstract syntax tree module to manipulate and analyze Python abstract syntax grammatically. Documentation: https://docs.python.org/3/library/ast.html
import json  # Importing the JSON module to encode and decode JSON data. Documentation: https://docs.python.org/3/library/json.html
import tkinter as tk  # Importing the tkinter module for creating graphical user interfaces. Documentation: https://docs.python.org/3/library/tkinter.html
from tkinter import (
import os  # Importing the os module to interact with the operating system. Documentation: https://docs.python.org/3/library/os.html
import logging  # Importing the logging module to enable logging capabilities. Documentation: https://docs.python.org/3/library/logging.html
from typing import (
import docstring_parser  # Importing the docstring_parser module to parse Python docstrings. Documentation: https://pypi.org/project/docstring-parser/
from concurrent.futures import (
def _extract_decorators(self, node: ast.FunctionDef) -> List[str]:
    """
        Extracts and infers types of decorators applied to functions and classes.

        Parameters:
            node (ast.FunctionDef): The node from which decorators are to be extracted.

        Returns:
            List[str]: A list of inferred decorator types.

        This method logs the extracted decorators and handles both simple and complex decorators.
        """
    decorators = [self._infer_decorator_type(decorator) for decorator in node.decorator_list]
    logging.debug(f'Extracted decorators for {node.name}: {decorators}')
    return decorators
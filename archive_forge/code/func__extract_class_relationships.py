import ast  # Importing the abstract syntax tree module to manipulate and analyze Python abstract syntax grammatically. Documentation: https://docs.python.org/3/library/ast.html
import json  # Importing the JSON module to encode and decode JSON data. Documentation: https://docs.python.org/3/library/json.html
import tkinter as tk  # Importing the tkinter module for creating graphical user interfaces. Documentation: https://docs.python.org/3/library/tkinter.html
from tkinter import (
import os  # Importing the os module to interact with the operating system. Documentation: https://docs.python.org/3/library/os.html
import logging  # Importing the logging module to enable logging capabilities. Documentation: https://docs.python.org/3/library/logging.html
from typing import (
import docstring_parser  # Importing the docstring_parser module to parse Python docstrings. Documentation: https://pypi.org/project/docstring-parser/
from concurrent.futures import (
def _extract_class_relationships(self, node: ast.ClassDef) -> List[str]:
    """
        Extracts and infers types of base classes for a given class definition.

        Parameters:
            node (ast.ClassDef): The class definition node from which base classes are to be extracted.

        Returns:
            List[str]: A list of inferred types of base classes.

        This method logs the extracted class relationships and handles the inference of base class types.
        """
    bases = [self._infer_complex_type(base) for base in node.bases]
    logging.debug(f'Extracted class relationships for {node.name}: {bases}')
    return bases
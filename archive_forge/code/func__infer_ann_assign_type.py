import ast
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import logging
from typing import List, Dict, Any, Optional, Union
import docstring_parser
def _infer_ann_assign_type(self, assign: ast.AnnAssign) -> str:
    """
        Helper method to infer type from an AnnAssign node.

        Parameters:
            assign (ast.AnnAssign): The AnnAssign node from AST.

        Returns:
            str: The inferred type as a string.
        """
    if isinstance(assign.annotation, ast.Name):
        return assign.annotation.id
    elif isinstance(assign.annotation, ast.Attribute):
        return f'{assign.annotation.value.id}.{assign.annotation.attr}'
    elif isinstance(assign.annotation, ast.Subscript):
        base = self.infer_type(assign.annotation.value)
        if isinstance(assign.annotation.slice, ast.Index):
            index = self.infer_type(assign.annotation.slice.value)
            return f'{base}[{index}]'
    return 'Unknown'
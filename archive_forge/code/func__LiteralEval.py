from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import ast
def _LiteralEval(value):
    """Parse value as a Python literal, or container of containers and literals.

  First the AST of the value is updated so that bare-words are turned into
  strings. Then the resulting AST is evaluated as a literal or container of
  only containers and literals.

  This allows for the YAML-like syntax {a: b} to represent the dict {'a': 'b'}

  Args:
    value: A string to be parsed as a literal or container of containers and
      literals.
  Returns:
    The Python value representing the value arg.
  Raises:
    ValueError: If the value is not an expression with only containers and
      literals.
    SyntaxError: If the value string has a syntax error.
  """
    root = ast.parse(value, mode='eval')
    if isinstance(root.body, ast.BinOp):
        raise ValueError(value)
    for node in ast.walk(root):
        for field, child in ast.iter_fields(node):
            if isinstance(child, list):
                for index, subchild in enumerate(child):
                    if isinstance(subchild, ast.Name):
                        child[index] = _Replacement(subchild)
            elif isinstance(child, ast.Name):
                replacement = _Replacement(child)
                setattr(node, field, replacement)
    return ast.literal_eval(root)
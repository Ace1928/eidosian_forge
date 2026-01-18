import argparse
import ast
import re
import sys
def _parse_slices(slicing_string):
    """Construct a tuple of slices from the slicing string.

  The string must be a valid slicing string.

  Args:
    slicing_string: (str) Input slicing string to be parsed.

  Returns:
    tuple(slice1, slice2, ...)

  Raises:
    ValueError: If tensor_slicing is not a valid numpy ndarray slicing str.
  """
    parsed = []
    for slice_string in slicing_string[1:-1].split(','):
        indices = slice_string.split(':')
        if len(indices) == 1:
            parsed.append(int(indices[0].strip()))
        elif 2 <= len(indices) <= 3:
            parsed.append(slice(*[int(index.strip()) if index.strip() else None for index in indices]))
        else:
            raise ValueError('Invalid tensor-slicing string.')
    return tuple(parsed)
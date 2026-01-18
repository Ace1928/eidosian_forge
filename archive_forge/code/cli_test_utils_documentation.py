import re
import numpy as np
Assert that the array value represented by lines is close to expected.

  Note that the shape of the array represented by the `array_lines` is ignored.

  Args:
    test: An instance of TensorFlowTestCase.
    expected_array: Expected value of the array.
    array_lines: A list of strings representing the array.
      E.g., "array([[ 1.0, 2.0 ], [ 3.0, 4.0 ]])"
      Assumes that values are separated by commas, parentheses, brackets, "|"
      characters and whitespace.
  
import doctest
import re
import textwrap
import numpy as np
def _tf_tensor_numpy_output(self, string):
    modified_string = self._NUMPY_OUTPUT_RE.sub('\\1', string)
    return (modified_string, modified_string != string)
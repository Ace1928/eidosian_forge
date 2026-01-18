import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
def _find_examples(name):
    return os.path.join(str(tmp_module), 'examples')
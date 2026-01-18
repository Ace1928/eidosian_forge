from typing import Optional, Iterator, Tuple, List
from parso.python.tokenize import tokenize
from parso.utils import parse_version_string
from parso.python.token import PythonTokenTypes
def _gettoken(self):
    tup = next(self.generator)
    self.type, self.value, self.begin, prefix = tup
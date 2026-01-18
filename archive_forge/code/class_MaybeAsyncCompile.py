import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
class MaybeAsyncCompile(Compile):

    def __init__(self, extra_flags=0):
        super().__init__()
        self.flags |= extra_flags
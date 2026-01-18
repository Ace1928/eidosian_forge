import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
class assemble_python_lines(TokenInputTransformer):

    def __init__(self):
        super(assemble_python_lines, self).__init__(None)

    def output(self, tokens):
        return self.reset()
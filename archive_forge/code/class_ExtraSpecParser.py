from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
class ExtraSpecParser(SpecParser):
    """
    Extension of SpecParser that works on extra .pythran files
    """

    def __call__(self, text, input_file=None):
        text = re.sub('^\\s*export', '#pythran export', text, flags=re.MULTILINE)
        return super(ExtraSpecParser, self).__call__(text, input_file)
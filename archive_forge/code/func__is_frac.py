import re
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
from ochat.evaluation.grading import math_normalize
def _is_frac(expr: str) -> bool:
    return bool(re.search('^-?[0-9]+.?/0*[1-9][0-9]*.?$', expr))
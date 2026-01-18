import re
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
from ochat.evaluation.grading import math_normalize
def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-07
    except:
        return False
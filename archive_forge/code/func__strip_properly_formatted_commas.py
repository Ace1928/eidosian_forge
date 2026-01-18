import re
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
from ochat.evaluation.grading import math_normalize
def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile('(\\d)(,)(\\d\\d\\d)($|\\D)')
    while True:
        next_expr = p1.sub('\\1\\3\\4', expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr
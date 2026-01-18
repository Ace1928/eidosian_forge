import re
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
from ochat.evaluation.grading import math_normalize
def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace('\\tfrac', '\\frac')
    expr = expr.replace('\\dfrac', '\\frac')
    expr = expr.replace('\\frac', ' \\frac')
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = expr.replace('√', 'sqrt')
    expr = expr.replace('π', 'pi')
    expr = expr.replace('∞', 'inf')
    expr = expr.replace('∪', 'U')
    expr = expr.replace('·', '*')
    expr = expr.replace('×', '*')
    return expr.strip()
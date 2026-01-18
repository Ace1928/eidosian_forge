from .numbers import number_to_scientific_html
from .string import StrPrinter
def _html_clsname(key):
    return 'chempy_' + key.replace('+', 'plus').replace('-', 'minus').replace('(', 'leftparen').replace(')', 'rightparen')
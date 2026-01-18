from collections import OrderedDict
from ..chemistry import Substance
from .numbers import number_to_scientific_html
def _elem(k):
    try:
        return cont[k]
    except (IndexError, TypeError):
        return cont[list(substances.keys()).index(k)]
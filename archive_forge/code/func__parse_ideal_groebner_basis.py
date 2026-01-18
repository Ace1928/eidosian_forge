from .polynomial import Polynomial
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
from . import processFileBase
from . import utilities
import snappy
import re
import sys
import tempfile
import subprocess
import shutil
def _parse_ideal_groebner_basis(text, py_eval, manifold_thunk, free_vars, witnesses, genus):
    match = re.match('Ideal of Polynomial ring of rank.*?\\n\\s*?(Order:\\s*?(.*?)|(.*?)\\s*?Order)\\n\\s*?Variables:(.*?\\n)+.*?Dimension (\\d+).*?\\s*([^,]*[Pp]rime)?.*?\\n(\\s*?Size of variety over algebraically closed field: (\\d+).*?\\n)?\\s*Groebner basis:\\n\\s*?\\[([^\\[\\]]*)\\]$', text)
    if not match:
        raise ValueError('Parsing error in component of decomposition: %s' % text)
    tot_order_str, post_order_str, pre_order_str, var_str, dimension_str, prime_str, variety_str, size_str, poly_strs = match.groups()
    dimension = int(dimension_str)
    if dimension == 0:
        polys = [Polynomial.parse_string(p) for p in poly_strs.replace('\n', ' ').split(',')]
    else:
        polys = []
    order_str = post_order_str if post_order_str else pre_order_str
    if not order_str:
        raise ValueError('Could not parse order in decomposition')
    if order_str.strip().lower() == 'lexicographical':
        term_order = 'lex'
    else:
        term_order = 'other'
    is_prime = prime_str is None or prime_str.lower() == 'prime'
    return PtolemyVarietyPrimeIdealGroebnerBasis(polys=polys, term_order=term_order, size=processFileBase.parse_int_or_empty(size_str), dimension=dimension, is_prime=is_prime, free_variables=free_vars, py_eval=py_eval, manifold_thunk=manifold_thunk, witnesses=witnesses, genus=genus)
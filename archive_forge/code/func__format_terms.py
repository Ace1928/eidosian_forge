import numbers
from typing import (
from typing_extensions import Self
def _format_terms(terms: Iterable[Tuple[TVector, Scalar]], format_spec: str):
    formatted_terms = [_format_term(format_spec, vector, coeff) for vector, coeff in terms]
    s = ''.join(formatted_terms)
    if not s:
        return f'{0:{format_spec}}'
    if s[0] == '+':
        return s[1:]
    return s
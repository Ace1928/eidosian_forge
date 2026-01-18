import numbers
from typing import (
from typing_extensions import Self
def _format_coefficient(format_spec: str, coefficient: Scalar) -> str:
    coefficient = complex(coefficient)
    real_str = f'{coefficient.real:{format_spec}}'
    imag_str = f'{coefficient.imag:{format_spec}}'
    if float(real_str) == 0 and float(imag_str) == 0:
        return ''
    if float(imag_str) == 0:
        return real_str
    if float(real_str) == 0:
        return imag_str + 'j'
    if real_str[0] == '-' and imag_str[0] == '-':
        return f'-({real_str[1:]}+{imag_str[1:]}j)'
    if imag_str[0] in ['+', '-']:
        return f'({real_str}{imag_str}j)'
    return f'({real_str}+{imag_str}j)'
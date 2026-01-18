from typing import (
import cmath
import re
import numpy as np
import sympy
def _translate_token(token_id: str, token_map: Mapping[str, _HangingToken]) -> _HangingToken:
    if re.match('[0-9]+(\\.[0-9]+)?', token_id):
        return float(token_id)
    if token_id in token_map:
        return token_map[token_id]
    raise ValueError(f'Unrecognized token: {token_id}')
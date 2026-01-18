import re
from fractions import Fraction
from typing import (
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq import protocols, value
from cirq._doc import doc_private
def _is_exposed_formula(text: str) -> bool:
    return re.match('[a-zA-Z_][a-zA-Z0-9_]*$', text) is None
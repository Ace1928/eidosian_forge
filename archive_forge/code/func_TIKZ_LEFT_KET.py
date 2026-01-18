from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Mapping, Optional, Set, Tuple, cast
from warnings import warn
from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
def TIKZ_LEFT_KET(qubit: int) -> str:
    return '\\lstick{{\\ket{{q_{{{qubit}}}}}}}'.format(qubit=qubit)
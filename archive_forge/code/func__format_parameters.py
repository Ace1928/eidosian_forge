from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Mapping, Optional, Set, Tuple, cast
from warnings import warn
from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
def _format_parameters(params: Iterable[ParameterDesignator], settings: Optional[DiagramSettings]=None) -> str:
    return '(' + ','.join((_format_parameter(param, settings) for param in params)) + ')'
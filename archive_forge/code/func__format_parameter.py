from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Mapping, Optional, Set, Tuple, cast
from warnings import warn
from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
def _format_parameter(param: ParameterDesignator, settings: Optional[DiagramSettings]=None) -> str:
    formatted = format_parameter(param)
    if settings and settings.texify_numerical_constants:
        formatted = formatted.replace('pi', '\\pi')
    return formatted
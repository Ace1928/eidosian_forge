from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Mapping, Optional, Set, Tuple, cast
from warnings import warn
from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
def TIKZ_GATE(name: str, size: int=1, params: Optional[Sequence[ParameterDesignator]]=None, dagger: bool=False, settings: Optional[DiagramSettings]=None) -> str:
    cmd = '\\gate'
    rotations = ['RX', 'RY', 'RZ']
    if settings and settings.abbreviate_controlled_rotations and (name in rotations) and params:
        name = name[1] + '_{{{param}}}'.format(param=_format_parameter(params[0], settings))
        return cmd + '{{{name}}}'.format(name=name)
    if size > 1:
        cmd += '[wires={size}]'.format(size=size)
    if name in ['RX', 'RY', 'RZ']:
        name = name[0] + '_' + name[1].lower()
    if dagger:
        name += '^{\\dagger}'
    if params:
        name += _format_parameters(params, settings)
    return cmd + '{{{name}}}'.format(name=name)
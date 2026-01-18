import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class DefWaveform(AbstractInstruction):

    def __init__(self, name: str, parameters: List[Parameter], entries: List[Union[Complex, Expression]]):
        self.name = name
        self.parameters = parameters
        self.entries = entries
        for e in entries:
            if not isinstance(e, (Complex, Expression)):
                raise TypeError(f'Unsupported waveform entry {e}')

    def out(self) -> str:
        ret = f'DEFWAVEFORM {self.name}'
        if len(self.parameters) > 0:
            first_param, *params = self.parameters
            ret += f'({first_param}'
            for param in params:
                ret += f', {param}'
            ret += ')'
        ret += ':\n    '
        ret += ', '.join(map(_complex_str, self.entries))
        return ret
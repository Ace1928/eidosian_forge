import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class JumpTarget(AbstractInstruction):
    """
    Representation of a target that can be jumped to.
    """

    def __init__(self, label: Union[Label, LabelPlaceholder]):
        if not isinstance(label, (Label, LabelPlaceholder)):
            raise TypeError('label must be a Label')
        self.label = label

    def __repr__(self) -> str:
        return '<JumpTarget {0}>'.format(str(self.label))

    def out(self) -> str:
        return 'LABEL {0}'.format(str(self.label))
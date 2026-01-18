import datetime
import math
import typing as t
from wandb.util import (
class InvalidType(Type):
    """A disallowed type.

    Assignments to a InvalidType result in a Never Type. InvalidType is basically the
    invalid case.
    """
    name = 'invalid'
    types: t.ClassVar[t.List[type]] = []

    def assign_type(self, wb_type: 'Type') -> 'InvalidType':
        return self
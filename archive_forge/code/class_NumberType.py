import datetime
import math
import typing as t
from wandb.util import (
class NumberType(Type):
    name = 'number'
    types: t.ClassVar[t.List[type]] = [int, float]
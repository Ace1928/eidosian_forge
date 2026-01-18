import datetime
import math
import typing as t
from wandb.util import (
class BooleanType(Type):
    name = 'boolean'
    types: t.ClassVar[t.List[type]] = [bool]
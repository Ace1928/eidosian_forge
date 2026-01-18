import enum
from typing import Optional, List, Union, Iterable, Tuple
class DurationLiteral(Expression):

    def __init__(self, value: float, unit: DurationUnit):
        self.value = value
        self.unit = unit
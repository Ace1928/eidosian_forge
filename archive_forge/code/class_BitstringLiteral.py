import enum
from typing import Optional, List, Union, Iterable, Tuple
class BitstringLiteral(Expression):

    def __init__(self, value, width):
        self.value = value
        self.width = width
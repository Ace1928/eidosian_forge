import enum
from typing import Optional, List, Union, Iterable, Tuple
class StringifyAndPray(Expression):

    def __init__(self, obj):
        self.obj = obj
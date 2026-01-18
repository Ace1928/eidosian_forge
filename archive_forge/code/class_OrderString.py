from abc import ABC, abstractmethod
from typing import TypeVar, Union
class OrderString(TypeValidator):

    def __init__(self):
        super().__init__(attr_type=str)

    def call(self, attr_name, value):
        super().call(attr_name, value)
        if value[0] not in {'+', '-'}:
            raise ValueError(f'{attr_name} must be prefixed with "+" or "-" to indicate ascending or descending order')
from abc import ABC, abstractmethod
from typing import TypeVar, Union
class Between(Validator):

    def __init__(self, lb, ub, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lb = lb
        self.ub = ub

    def call(self, attr_name, value):
        if not self.lb <= value <= self.ub:
            raise ValueError(f'{attr_name} must be between [{self.lb}, {self.ub}] inclusive (got {value})')
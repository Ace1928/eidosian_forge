import pytest
from ase.build import molecule
from ase.calculators.calculator import get_calculator_class
from ase.units import Ry
from ase.utils import workdir
class CalculatorInputs:

    def __init__(self, name, parameters=None):
        self.name = name
        if parameters is None:
            parameters = {}
        self.parameters = parameters

    def __repr__(self):
        cls = type(self)
        return '{}({}, {})'.format(cls.__name__, self.name, self.parameters)

    def calc(self):
        cls = get_calculator_class(self.name)
        return cls(**self.parameters)
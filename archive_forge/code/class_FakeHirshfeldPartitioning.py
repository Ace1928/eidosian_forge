from ase import io
from ase.calculators.vdwcorrection import vdWTkatchenko09prl
from ase.calculators.emt import EMT
from ase.build import bulk
class FakeHirshfeldPartitioning:

    def __init__(self, calculator):
        self.calculator = calculator

    def initialize(self):
        pass

    def get_effective_volume_ratios(self):
        return [1]

    def get_calculator(self):
        return self.calculator
from ase import io
from ase.calculators.vdwcorrection import vdWTkatchenko09prl
from ase.calculators.emt import EMT
from ase.build import bulk
class FakeDFTcalculator(EMT):

    def get_xc_functional(self):
        return 'PBE'
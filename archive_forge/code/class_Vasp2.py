from ase.utils import deprecated
from .vasp import Vasp
class Vasp2(Vasp):

    @deprecated('Vasp2 has been deprecated. Use the ase.calculators.vasp.Vasp class instead.')
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
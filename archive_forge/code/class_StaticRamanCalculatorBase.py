import numpy as np
import ase.units as u
from ase.parallel import world
from ase.phonons import Phonons
from ase.vibrations.vibrations import Vibrations, AtomicDisplacements
from ase.dft import monkhorst_pack
from ase.utils import IOContext
class StaticRamanCalculatorBase(RamanCalculatorBase):
    """Base class for Raman intensities derived from
    static polarizabilities"""

    def __init__(self, atoms, exobj, exkwargs=None, *args, **kwargs):
        self.exobj = exobj
        if exkwargs is None:
            exkwargs = {}
        self.exkwargs = exkwargs
        super().__init__(atoms, *args, **kwargs)

    def _new_exobj(self):
        return self.exobj(**self.exkwargs)

    def calculate(self, atoms, disp):
        returnvalue = super().calculate(atoms, disp)
        disp.calculate_and_save_static_polarizability(atoms)
        return returnvalue
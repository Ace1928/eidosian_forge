import sys
import numpy as np
import ase.units as u
from ase.parallel import world, paropen, parprint
from ase.vibrations import Vibrations
from ase.vibrations.raman import Raman, RamanCalculatorBase
class ResonantRamanCalculator(RamanCalculatorBase, Vibrations):
    """Base class for resonant Raman calculators using finite differences.
    """

    def __init__(self, atoms, ExcitationsCalculator, *args, exkwargs=None, exext='.ex.gz', overlap=False, **kwargs):
        """
        Parameters
        ----------
        atoms: Atoms
            The Atoms object
        ExcitationsCalculator: object
            Calculator for excited states
        exkwargs: dict
            Arguments given to the ExcitationsCalculator object
        exext: string
            Extension for filenames of Excitation lists (results of
            the ExcitationsCalculator).
        overlap : function or False
            Function to calculate overlaps between excitation at
            equilibrium and at a displaced position. Calculators are
            given as first and second argument, respectively.

        Example
        -------

        >>> from ase.calculators.h2morse import (H2Morse,
        ...                                      H2MorseExcitedStatesCalculator)
        >>> from ase.vibrations.resonant_raman import ResonantRamanCalculator
        >>>
        >>> atoms = H2Morse()
        >>> rmc = ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator)
        >>> rmc.run()

        This produces all necessary data for further analysis.
        """
        self.exobj = ExcitationsCalculator
        if exkwargs is None:
            exkwargs = {}
        self.exkwargs = exkwargs
        self.overlap = overlap
        super().__init__(atoms, *args, exext=exext, **kwargs)

    def _new_exobj(self):
        return self.exobj(**self.exkwargs)

    def calculate(self, atoms, disp):
        """Call ground and excited state calculation"""
        assert atoms == self.atoms
        forces = self.atoms.get_forces()
        if self.overlap:
            'Overlap is determined as\n\n            ov_ij = int dr displaced*_i(r) eqilibrium_j(r)\n            '
            ov_nn = self.overlap(self.atoms.calc, self.eq_calculator)
            if world.rank == 0:
                disp.save_ov_nn(ov_nn)
        disp.calculate_and_save_exlist(atoms)
        return {'forces': forces}

    def run(self):
        if self.overlap:
            self.atoms.get_potential_energy()
            self.eq_calculator = self.atoms.calc
            fname = 'tmp.gpw'
            self.eq_calculator.write(fname, 'all')
            self.eq_calculator = self.eq_calculator.__class__(restart=fname)
            try:
                self.eq_calculator.converge_wave_functions()
            except AttributeError:
                pass
        Vibrations.run(self)
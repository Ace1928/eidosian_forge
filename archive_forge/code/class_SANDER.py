import subprocess
import numpy as np
from ase.calculators.calculator import Calculator, FileIOCalculator
import ase.units as units
from scipy.io import netcdf
class SANDER(Calculator):
    """
    Interface to SANDER using Python interface

    Requires sander Python bindings from http://ambermd.org/
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, atoms=None, label=None, top=None, crd=None, mm_options=None, qm_options=None, permutation=None, **kwargs):
        if not have_sander:
            raise RuntimeError('sander Python module could not be imported!')
        Calculator.__init__(self, label, atoms)
        self.permutation = permutation
        if qm_options is not None:
            sander.setup(top, crd.coordinates, crd.box, mm_options, qm_options)
        else:
            sander.setup(top, crd.coordinates, crd.box, mm_options)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if system_changes:
            if 'energy' in self.results:
                del self.results['energy']
            if 'forces' in self.results:
                del self.results['forces']
        if 'energy' not in self.results:
            if self.permutation is None:
                crd = np.reshape(atoms.get_positions(), (1, len(atoms), 3))
            else:
                crd = np.reshape(atoms.get_positions()[self.permutation[0, :]], (1, len(atoms), 3))
            sander.set_positions(crd)
            e, f = sander.energy_forces()
            self.results['energy'] = e.tot * units.kcal / units.mol
            if self.permutation is None:
                self.results['forces'] = np.reshape(np.array(f), (len(atoms), 3)) * units.kcal / units.mol
            else:
                ff = np.reshape(np.array(f), (len(atoms), 3)) * units.kcal / units.mol
                self.results['forces'] = ff[self.permutation[1, :]]
        if 'forces' not in self.results:
            if self.permutation is None:
                crd = np.reshape(atoms.get_positions(), (1, len(atoms), 3))
            else:
                crd = np.reshape(atoms.get_positions()[self.permutation[0, :]], (1, len(atoms), 3))
            sander.set_positions(crd)
            e, f = sander.energy_forces()
            self.results['energy'] = e.tot * units.kcal / units.mol
            if self.permutation is None:
                self.results['forces'] = np.reshape(np.array(f), (len(atoms), 3)) * units.kcal / units.mol
            else:
                ff = np.reshape(np.array(f), (len(atoms), 3)) * units.kcal / units.mol
                self.results['forces'] = ff[self.permutation[1, :]]
import numpy as np
from ase.calculators.calculator import Calculator, all_properties
from ase.calculators.calculator import PropertyNotImplementedError
class SinglePointCalculator(Calculator):
    """Special calculator for a single configuration.

    Used to remember the energy, force and stress for a given
    configuration.  If the positions, atomic numbers, unit cell, or
    boundary conditions are changed, then asking for
    energy/forces/stress will raise an exception."""
    name = 'unknown'

    def __init__(self, atoms, **results):
        """Save energy, forces, stress, ... for the current configuration."""
        Calculator.__init__(self)
        self.results = {}
        for property, value in results.items():
            assert property in all_properties
            if value is None:
                continue
            if property in ['energy', 'magmom', 'free_energy']:
                self.results[property] = value
            else:
                self.results[property] = np.array(value, float)
        self.atoms = atoms.copy()

    def __str__(self):
        tokens = []
        for key, val in sorted(self.results.items()):
            if np.isscalar(val):
                txt = '{}={}'.format(key, val)
            else:
                txt = '{}=...'.format(key)
            tokens.append(txt)
        return '{}({})'.format(self.__class__.__name__, ', '.join(tokens))

    def get_property(self, name, atoms=None, allow_calculation=True):
        if atoms is None:
            atoms = self.atoms
        if name not in self.results or self.check_state(atoms):
            if allow_calculation:
                raise PropertyNotImplementedError('The property "{0}" is not available.'.format(name))
            return None
        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result
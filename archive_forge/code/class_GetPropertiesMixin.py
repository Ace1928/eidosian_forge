from abc import ABC, abstractmethod
from typing import Mapping, Any
class GetPropertiesMixin(ABC):
    """Mixin class which provides get_forces(), get_stress() and so on.

    Inheriting class must implement get_property()."""

    @abstractmethod
    def get_property(self, name, atoms=None, allow_calculation=True):
        """Get the named property."""

    def get_potential_energies(self, atoms=None):
        return self.get_property('energies', atoms)

    def get_forces(self, atoms=None):
        return self.get_property('forces', atoms)

    def get_stress(self, atoms=None):
        return self.get_property('stress', atoms)

    def get_stresses(self, atoms=None):
        """the calculator should return intensive stresses, i.e., such that
                stresses.sum(axis=0) == stress
        """
        return self.get_property('stresses', atoms)

    def get_dipole_moment(self, atoms=None):
        return self.get_property('dipole', atoms)

    def get_charges(self, atoms=None):
        return self.get_property('charges', atoms)

    def get_magnetic_moment(self, atoms=None):
        return self.get_property('magmom', atoms)

    def get_magnetic_moments(self, atoms=None):
        """Calculate magnetic moments projected onto atoms."""
        return self.get_property('magmoms', atoms)
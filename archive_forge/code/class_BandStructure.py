import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.calculators.calculator import PropertyNotImplementedError
@jsonable('bandstructure')
class BandStructure:
    """A band structure consists of an array of eigenvalues and a bandpath.

    BandStructure objects support JSON I/O.
    """

    def __init__(self, path, energies, reference=0.0):
        self._path = path
        self._energies = np.asarray(energies)
        assert self.energies.shape[0] in [1, 2]
        assert self.energies.shape[1] == len(path.kpts)
        assert np.isscalar(reference)
        self._reference = reference

    @property
    def energies(self) -> np.ndarray:
        """The energies of this band structure.

        This is a numpy array of shape (nspins, nkpoints, nbands)."""
        return self._energies

    @property
    def path(self) -> 'ase.dft.kpoints.BandPath':
        """The :class:`~ase.dft.kpoints.BandPath` of this band structure."""
        return self._path

    @property
    def reference(self) -> float:
        """The reference energy.

        Semantics may vary; typically a Fermi energy or zero,
        depending on how the band structure was created."""
        return self._reference

    def subtract_reference(self) -> 'BandStructure':
        """Return new band structure with reference energy subtracted."""
        return BandStructure(self.path, self.energies - self.reference, reference=0.0)

    def todict(self):
        return dict(path=self.path, energies=self.energies, reference=self.reference)

    def get_labels(self, eps=1e-05):
        """"See :func:`ase.dft.kpoints.labels_from_kpts`."""
        return self.path.get_linear_kpoint_axis(eps=eps)

    def plot(self, *args, **kwargs):
        """Plot this band structure."""
        bsp = BandStructurePlot(self)
        return bsp.plot(*args, **kwargs)

    def __repr__(self):
        return '{}(path={!r}, energies=[{} values], reference={})'.format(self.__class__.__name__, self.path, '{}x{}x{}'.format(*self.energies.shape), self.reference)
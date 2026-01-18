import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
@jsonable('bandpath')
class BandPath:
    """Represents a Brillouin zone path or bandpath.

    A band path has a unit cell, a path specification, special points,
    and interpolated k-points.  Band paths are typically created
    indirectly using the :class:`~ase.geometry.Cell` or
    :class:`~ase.lattice.BravaisLattice` classes:

    >>> from ase.lattice import CUB
    >>> path = CUB(3).bandpath()
    >>> path
    BandPath(path='GXMGRX,MR', cell=[3x3], special_points={GMRX}, kpts=[40x3])

    Band paths support JSON I/O:

    >>> from ase.io.jsonio import read_json
    >>> path.write('mybandpath.json')
    >>> read_json('mybandpath.json')
    BandPath(path='GXMGRX,MR', cell=[3x3], special_points={GMRX}, kpts=[40x3])

    """

    def __init__(self, cell, kpts=None, special_points=None, path=None):
        if kpts is None:
            kpts = np.empty((0, 3))
        if special_points is None:
            special_points = {}
        else:
            special_points = normalize_special_points(special_points)
        if path is None:
            path = ''
        cell = Cell(cell)
        self._cell = cell
        kpts = np.asarray(kpts)
        assert kpts.ndim == 2 and kpts.shape[1] == 3 and (kpts.dtype == float)
        self._icell = self.cell.reciprocal()
        self._kpts = kpts
        self._special_points = special_points
        if not isinstance(path, str):
            raise TypeError(f'path must be a string; was {path!r}')
        self._path = path

    @property
    def cell(self) -> Cell:
        """The :class:`~ase.cell.Cell` of this BandPath."""
        return self._cell

    @property
    def icell(self) -> Cell:
        """Reciprocal cell of this BandPath as a :class:`~ase.cell.Cell`."""
        return self._icell

    @property
    def kpts(self) -> np.ndarray:
        """The kpoints of this BandPath as an array of shape (nkpts, 3).

        The kpoints are given in units of the reciprocal cell."""
        return self._kpts

    @property
    def special_points(self) -> Dict[str, np.ndarray]:
        """Special points of this BandPath as a dictionary.

        The dictionary maps names (such as `'G'`) to kpoint coordinates
        in units of the reciprocal cell as a 3-element numpy array.

        It's unwise to edit this dictionary directly.  If you need that,
        consider deepcopying it."""
        return self._special_points

    @property
    def path(self) -> str:
        """The string specification of this band path.

        This is a specification of the form `'GXWKGLUWLK,UX'`.

        Comma marks a discontinuous jump: K is not connected to U."""
        return self._path

    def transform(self, op: np.ndarray) -> 'BandPath':
        """Apply 3x3 matrix to this BandPath and return new BandPath.

        This is useful for converting the band path to another cell.
        The operation will typically be a permutation/flipping
        established by a function such as Niggli reduction."""
        special_points = {}
        for name, value in self.special_points.items():
            special_points[name] = value @ op
        return BandPath(op.T @ self.cell, kpts=self.kpts @ op, special_points=special_points, path=self.path)

    def todict(self):
        return {'kpts': self.kpts, 'special_points': self.special_points, 'labelseq': self.path, 'cell': self.cell}

    def interpolate(self, path: str=None, npoints: int=None, special_points: Dict[str, np.ndarray]=None, density: float=None) -> 'BandPath':
        """Create new bandpath, (re-)interpolating kpoints from this one."""
        if path is None:
            path = self.path
        if special_points is None:
            special_points = self.special_points
        pathnames, pathcoords = resolve_kpt_path_string(path, special_points)
        kpts, x, X = paths2kpts(pathcoords, self.cell, npoints, density)
        return BandPath(self.cell, kpts, path=path, special_points=special_points)

    def _scale(self, coords):
        return np.dot(coords, self.icell)

    def __repr__(self):
        return '{}(path={}, cell=[3x3], special_points={{{}}}, kpts=[{}x3])'.format(self.__class__.__name__, repr(self.path), ''.join(sorted(self.special_points)), len(self.kpts))

    def cartesian_kpts(self) -> np.ndarray:
        """Get Cartesian kpoints from this bandpath."""
        return self._scale(self.kpts)

    def __iter__(self):
        """XXX Compatibility hack for bandpath() function.

        bandpath() now returns a BandPath object, which is a Good
        Thing.  However it used to return a tuple of (kpts, x_axis,
        special_x_coords), and people would use tuple unpacking for
        those.

        This function makes tuple unpacking work in the same way.
        It will be removed in the future.

        """
        import warnings
        warnings.warn('Please do not use (kpts, x, X) = bandpath(...).  Use path = bandpath(...) and then kpts = path.kpts and (x, X, labels) = path.get_linear_kpoint_axis().')
        yield self.kpts
        x, xspecial, _ = self.get_linear_kpoint_axis()
        yield x
        yield xspecial

    def __getitem__(self, index):
        return tuple(self)[index]

    def get_linear_kpoint_axis(self, eps=1e-05):
        """Define x axis suitable for plotting a band structure.

        See :func:`ase.dft.kpoints.labels_from_kpts`."""
        index2name = self._find_special_point_indices(eps)
        indices = sorted(index2name)
        labels = [index2name[index] for index in indices]
        xcoords, special_xcoords = indices_to_axis_coords(indices, self.kpts, self.cell)
        return (xcoords, special_xcoords, labels)

    def _find_special_point_indices(self, eps):
        """Find indices of kpoints which are close to special points.

        The result is returned as a dictionary mapping indices to labels."""
        index2name = {}
        nkpts = len(self.kpts)
        for name, kpt in self.special_points.items():
            displacements = self.kpts - kpt[np.newaxis, :]
            distances = np.linalg.norm(displacements, axis=1)
            args = np.argwhere(distances < eps)
            for arg in args.flat:
                dist = distances[arg]
                neighbours = distances[max(arg - 1, 0):min(arg + 1, nkpts - 1)]
                if not any(neighbours < dist):
                    index2name[arg] = name
        return index2name

    def plot(self, **plotkwargs):
        """Visualize this bandpath.

        Plots the irreducible Brillouin zone and this bandpath."""
        import ase.dft.bz as bz
        plotkwargs.pop('dimension', None)
        special_points = self.special_points
        labelseq, coords = resolve_kpt_path_string(self.path, special_points)
        paths = []
        points_already_plotted = set()
        for subpath_labels, subpath_coords in zip(labelseq, coords):
            subpath_coords = np.array(subpath_coords)
            points_already_plotted.update(subpath_labels)
            paths.append((subpath_labels, self._scale(subpath_coords)))
        for label, point in special_points.items():
            if label not in points_already_plotted:
                paths.append(([label], [self._scale(point)]))
        kw = {'vectors': True, 'pointstyle': {'marker': '.'}}
        kw.update(plotkwargs)
        return bz.bz_plot(self.cell, paths=paths, points=self.cartesian_kpts(), **kw)

    def free_electron_band_structure(self, **kwargs) -> 'ase.spectrum.band_structure.BandStructure':
        """Return band structure of free electrons for this bandpath.

        Keyword arguments are passed to
        :class:`~ase.calculators.test.FreeElectrons`.

        This is for mostly testing and visualization."""
        from ase import Atoms
        from ase.calculators.test import FreeElectrons
        from ase.spectrum.band_structure import calculate_band_structure
        atoms = Atoms(cell=self.cell, pbc=True)
        atoms.calc = FreeElectrons(**kwargs)
        bs = calculate_band_structure(atoms, path=self)
        return bs
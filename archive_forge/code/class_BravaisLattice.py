from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
class BravaisLattice(ABC):
    """Represent Bravais lattices and data related to the Brillouin zone.

    There are 14 3D Bravais classes: CUB, FCC, BCC, ..., and TRI, and
    five 2D classes.

    Each class stores basic static information:

    >>> from ase.lattice import FCC, MCL
    >>> FCC.name
    'FCC'
    >>> FCC.longname
    'face-centred cubic'
    >>> FCC.pearson_symbol
    'cF'
    >>> MCL.parameters
    ('a', 'b', 'c', 'alpha')

    Each class can be instantiated with the specific lattice parameters
    that apply to that lattice:

    >>> MCL(3, 4, 5, 80)
    MCL(a=3, b=4, c=5, alpha=80)

    """
    name = None
    longname = None
    parameters = None
    variants = None
    ndim = None

    def __init__(self, **kwargs):
        p = {}
        eps = kwargs.pop('eps', 0.0002)
        for k, v in kwargs.items():
            p[k] = float(v)
        assert set(p) == set(self.parameters)
        self._parameters = p
        self._eps = eps
        if len(self.variants) == 1:
            self._variant = self.variants[self.name]
        else:
            name = self._variant_name(**self._parameters)
            self._variant = self.variants[name]

    @property
    def variant(self) -> str:
        """Return name of lattice variant.

        >>> BCT(3, 5).variant
        'BCT2'
        """
        return self._variant.name

    def __getattr__(self, name: str):
        if name in self._parameters:
            return self._parameters[name]
        return self.__getattribute__(name)

    def vars(self) -> Dict[str, float]:
        """Get parameter names and values of this lattice as a dictionary."""
        return dict(self._parameters)

    def conventional(self) -> 'BravaisLattice':
        """Get the conventional cell corresponding to this lattice."""
        cls = bravais_lattices[self.conventional_cls]
        return cls(**self._parameters)

    def tocell(self) -> Cell:
        """Return this lattice as a :class:`~ase.cell.Cell` object."""
        cell = self._cell(**self._parameters)
        return Cell(cell)

    def get_transformation(self, cell, eps=1e-08):
        T = cell.dot(np.linalg.pinv(self.tocell()))
        msg = 'This transformation changes the length/area/volume of the cell'
        assert np.isclose(np.abs(np.linalg.det(T[:self.ndim, :self.ndim])), 1, atol=eps), msg
        return T

    def cellpar(self) -> np.ndarray:
        """Get cell lengths and angles as array of length 6.

        See :func:`ase.geometry.Cell.cellpar`."""
        cell = self.tocell()
        return cell.cellpar()

    @property
    def special_path(self) -> str:
        """Get default special k-point path for this lattice as a string.

        >>> BCT(3, 5).special_path
        'GXYSGZS1NPY1Z,XP'
        """
        return self._variant.special_path

    @property
    def special_point_names(self) -> List[str]:
        """Return all special point names as a list of strings.

        >>> BCT(3, 5).special_point_names
        ['G', 'N', 'P', 'S', 'S1', 'X', 'Y', 'Y1', 'Z']
        """
        labels = parse_path_string(self._variant.special_point_names)
        assert len(labels) == 1
        return labels[0]

    def get_special_points_array(self) -> np.ndarray:
        """Return all special points for this lattice as an array.

        Ordering is consistent with special_point_names."""
        if self._variant.special_points is not None:
            d = self.get_special_points()
            labels = self.special_point_names
            assert len(d) == len(labels)
            points = np.empty((len(d), 3))
            for i, label in enumerate(labels):
                points[i] = d[label]
            return points
        points = self._special_points(variant=self._variant, **self._parameters)
        assert len(points) == len(self.special_point_names)
        return np.array(points)

    def get_special_points(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of named special k-points for this lattice."""
        if self._variant.special_points is not None:
            return self._variant.special_points
        labels = self.special_point_names
        points = self.get_special_points_array()
        return dict(zip(labels, points))

    def plot_bz(self, path=None, special_points=None, **plotkwargs):
        """Plot the reciprocal cell and default bandpath."""
        bandpath = self.bandpath(path=path, special_points=special_points, npoints=0)
        return bandpath.plot(dimension=self.ndim, **plotkwargs)

    def bandpath(self, path=None, npoints=None, special_points=None, density=None, transformation=None) -> BandPath:
        """Return a :class:`~ase.dft.kpoints.BandPath` for this lattice.

        See :meth:`ase.cell.Cell.bandpath` for description of parameters.

        >>> BCT(3, 5).bandpath()
        BandPath(path='GXYSGZS1NPY1Z,XP', cell=[3x3], special_points={GNPSS1XYY1Z}, kpts=[51x3])

        .. note:: This produces the standard band path following AFlow
           conventions.  If your cell does not follow this convention,
           you will need :meth:`ase.cell.Cell.bandpath` instead or
           the kpoints may not correspond to your particular cell.

        """
        if special_points is None:
            special_points = self.get_special_points()
        if path is None:
            path = self._variant.special_path
        elif not isinstance(path, str):
            from ase.dft.kpoints import resolve_custom_points
            path, special_points = resolve_custom_points(path, special_points, self._eps)
        cell = self.tocell()
        if transformation is not None:
            cell = transformation.dot(cell)
        bandpath = BandPath(cell=cell, path=path, special_points=special_points)
        return bandpath.interpolate(npoints=npoints, density=density)

    @abstractmethod
    def _cell(self, **kwargs):
        """Return a Cell object from this Bravais lattice.

        Arguments are the dictionary of Bravais parameters."""
        pass

    def _special_points(self, **kwargs):
        """Return the special point coordinates as an npoints x 3 sequence.

        Subclasses typically return a nested list.

        Ordering must be same as kpoint labels.

        Arguments are the dictionary of Bravais parameters and the variant."""
        raise NotImplementedError

    def _variant_name(self, **kwargs):
        """Return the name (e.g. 'ORCF3') of variant.

        Arguments will be the dictionary of Bravais parameters."""
        raise NotImplementedError

    def __format__(self, spec):
        tokens = []
        if not spec:
            spec = '.6g'
        template = '{}={:%s}' % spec
        for name in self.parameters:
            value = self._parameters[name]
            tokens.append(template.format(name, value))
        return '{}({})'.format(self.name, ', '.join(tokens))

    def __str__(self) -> str:
        return self.__format__('')

    def __repr__(self) -> str:
        return self.__format__('.20g')

    def description(self) -> str:
        """Return complete description of lattice and Brillouin zone."""
        points = self.get_special_points()
        labels = self.special_point_names
        coordstring = '\n'.join(['    {:2s} {:7.4f} {:7.4f} {:7.4f}'.format(label, *points[label]) for label in labels])
        string = '{repr}\n  {variant}\n  Special point coordinates:\n{special_points}\n'.format(repr=str(self), variant=self._variant, special_points=coordstring)
        return string

    @classmethod
    def type_description(cls):
        """Return complete description of this Bravais lattice type."""
        desc = 'Lattice name: {name}\n  Long name: {longname}\n  Parameters: {parameters}\n'.format(**vars(cls))
        chunks = [desc]
        for name in cls.variant_names:
            var = cls.variants[name]
            txt = str(var)
            lines = ['  ' + L for L in txt.splitlines()]
            lines.append('')
            chunks.extend(lines)
        return '\n'.join(chunks)
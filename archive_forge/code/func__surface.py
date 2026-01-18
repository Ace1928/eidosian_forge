from math import sqrt
from operator import itemgetter
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.data import reference_states, atomic_numbers
from ase.lattice.cubic import FaceCenteredCubic
def _surface(symbol, structure, face, size, a, c, vacuum, periodic, orthogonal=True):
    """Function to build often used surfaces.

    Don't call this function directly - use fcc100, fcc110, bcc111, ..."""
    Z = atomic_numbers[symbol]
    if a is None:
        sym = reference_states[Z]['symmetry']
        if sym != structure:
            raise ValueError("Can't guess lattice constant for %s-%s!" % (structure, symbol))
        a = reference_states[Z]['a']
    if structure == 'hcp' and c is None:
        if reference_states[Z]['symmetry'] == 'hcp':
            c = reference_states[Z]['c/a'] * a
        else:
            c = sqrt(8 / 3.0) * a
    positions = np.empty((size[2], size[1], size[0], 3))
    positions[..., 0] = np.arange(size[0]).reshape((1, 1, -1))
    positions[..., 1] = np.arange(size[1]).reshape((1, -1, 1))
    positions[..., 2] = np.arange(size[2]).reshape((-1, 1, 1))
    numbers = np.ones(size[0] * size[1] * size[2], int) * Z
    tags = np.empty((size[2], size[1], size[0]), int)
    tags[:] = np.arange(size[2], 0, -1).reshape((-1, 1, 1))
    slab = Atoms(numbers, tags=tags.ravel(), pbc=(True, True, periodic), cell=size)
    surface_cell = None
    sites = {'ontop': (0, 0)}
    surf = structure + face
    if surf == 'fcc100':
        cell = (sqrt(0.5), sqrt(0.5), 0.5)
        positions[-2::-2, ..., :2] += 0.5
        sites.update({'hollow': (0.5, 0.5), 'bridge': (0.5, 0)})
    elif surf == 'diamond100':
        cell = (sqrt(0.5), sqrt(0.5), 0.5 / 2)
        positions[-4::-4, ..., :2] += (0.5, 0.5)
        positions[-3::-4, ..., :2] += (0.0, 0.5)
        positions[-2::-4, ..., :2] += (0.0, 0.0)
        positions[-1::-4, ..., :2] += (0.5, 0.0)
    elif surf == 'fcc110':
        cell = (1.0, sqrt(0.5), sqrt(0.125))
        positions[-2::-2, ..., :2] += 0.5
        sites.update({'hollow': (0.5, 0.5), 'longbridge': (0.5, 0), 'shortbridge': (0, 0.5)})
    elif surf == 'bcc100':
        cell = (1.0, 1.0, 0.5)
        positions[-2::-2, ..., :2] += 0.5
        sites.update({'hollow': (0.5, 0.5), 'bridge': (0.5, 0)})
    else:
        if orthogonal and size[1] % 2 == 1:
            raise ValueError("Can't make orthorhombic cell with size=%r.  " % (tuple(size),) + 'Second number in size must be even.')
        if surf == 'fcc111':
            cell = (sqrt(0.5), sqrt(0.375), 1 / sqrt(3))
            if orthogonal:
                positions[-1::-3, 1::2, :, 0] += 0.5
                positions[-2::-3, 1::2, :, 0] += 0.5
                positions[-3::-3, 1::2, :, 0] -= 0.5
                positions[-2::-3, ..., :2] += (0.0, 2.0 / 3)
                positions[-3::-3, ..., :2] += (0.5, 1.0 / 3)
            else:
                positions[-2::-3, ..., :2] += (-1.0 / 3, 2.0 / 3)
                positions[-3::-3, ..., :2] += (1.0 / 3, 1.0 / 3)
            sites.update({'bridge': (0.5, 0), 'fcc': (1.0 / 3, 1.0 / 3), 'hcp': (2.0 / 3, 2.0 / 3)})
        elif surf == 'diamond111':
            cell = (sqrt(0.5), sqrt(0.375), 1 / sqrt(3) / 2)
            assert not orthogonal
            positions[-1::-6, ..., :3] += (0.0, 0.0, 0.5)
            positions[-2::-6, ..., :2] += (0.0, 0.0)
            positions[-3::-6, ..., :3] += (-1.0 / 3, 2.0 / 3, 0.5)
            positions[-4::-6, ..., :2] += (-1.0 / 3, 2.0 / 3)
            positions[-5::-6, ..., :3] += (1.0 / 3, 1.0 / 3, 0.5)
            positions[-6::-6, ..., :2] += (1.0 / 3, 1.0 / 3)
        elif surf == 'hcp0001':
            cell = (1.0, sqrt(0.75), 0.5 * c / a)
            if orthogonal:
                positions[:, 1::2, :, 0] += 0.5
                positions[-2::-2, ..., :2] += (0.0, 2.0 / 3)
            else:
                positions[-2::-2, ..., :2] += (-1.0 / 3, 2.0 / 3)
            sites.update({'bridge': (0.5, 0), 'fcc': (1.0 / 3, 1.0 / 3), 'hcp': (2.0 / 3, 2.0 / 3)})
        elif surf == 'hcp10m10':
            cell = (1.0, 0.5 * c / a, sqrt(0.75))
            assert orthogonal
            positions[-2::-2, ..., 0] += 0.5
            positions[:, ::2, :, 2] += 2.0 / 3
        elif surf == 'bcc110':
            cell = (1.0, sqrt(0.5), sqrt(0.5))
            if orthogonal:
                positions[:, 1::2, :, 0] += 0.5
                positions[-2::-2, ..., :2] += (0.0, 1.0)
            else:
                positions[-2::-2, ..., :2] += (-0.5, 1.0)
            sites.update({'shortbridge': (0, 0.5), 'longbridge': (0.5, 0), 'hollow': (0.375, 0.25)})
        elif surf == 'bcc111':
            cell = (sqrt(2), sqrt(1.5), sqrt(3) / 6)
            if orthogonal:
                positions[-1::-3, 1::2, :, 0] += 0.5
                positions[-2::-3, 1::2, :, 0] += 0.5
                positions[-3::-3, 1::2, :, 0] -= 0.5
                positions[-2::-3, ..., :2] += (0.0, 2.0 / 3)
                positions[-3::-3, ..., :2] += (0.5, 1.0 / 3)
            else:
                positions[-2::-3, ..., :2] += (-1.0 / 3, 2.0 / 3)
                positions[-3::-3, ..., :2] += (1.0 / 3, 1.0 / 3)
            sites.update({'hollow': (1.0 / 3, 1.0 / 3)})
        else:
            2 / 0
        surface_cell = a * np.array([(cell[0], 0), (cell[0] / 2, cell[1])])
        if not orthogonal:
            cell = np.array([(cell[0], 0, 0), (cell[0] / 2, cell[1], 0), (0, 0, cell[2])])
    if surface_cell is None:
        surface_cell = a * np.diag(cell[:2])
    if isinstance(cell, tuple):
        cell = np.diag(cell)
    slab.set_positions(positions.reshape((-1, 3)))
    slab.set_cell([a * v * n for v, n in zip(cell, size)], scale_atoms=True)
    if not periodic:
        slab.cell[2] = 0.0
    if vacuum is not None:
        slab.center(vacuum, axis=2)
    if 'adsorbate_info' not in slab.info:
        slab.info.update({'adsorbate_info': {}})
    slab.info['adsorbate_info']['cell'] = surface_cell
    slab.info['adsorbate_info']['sites'] = sites
    return slab
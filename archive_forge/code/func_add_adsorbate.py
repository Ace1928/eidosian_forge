from math import sqrt
from operator import itemgetter
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.data import reference_states, atomic_numbers
from ase.lattice.cubic import FaceCenteredCubic
def add_adsorbate(slab, adsorbate, height, position=(0, 0), offset=None, mol_index=0):
    """Add an adsorbate to a surface.

    This function adds an adsorbate to a slab.  If the slab is
    produced by one of the utility functions in ase.build, it
    is possible to specify the position of the adsorbate by a keyword
    (the supported keywords depend on which function was used to
    create the slab).

    If the adsorbate is a molecule, the atom indexed by the mol_index
    optional argument is positioned on top of the adsorption position
    on the surface, and it is the responsibility of the user to orient
    the adsorbate in a sensible way.

    This function can be called multiple times to add more than one
    adsorbate.

    Parameters:

    slab: The surface onto which the adsorbate should be added.

    adsorbate:  The adsorbate. Must be one of the following three types:
        A string containing the chemical symbol for a single atom.
        An atom object.
        An atoms object (for a molecular adsorbate).

    height: Height above the surface.

    position: The x-y position of the adsorbate, either as a tuple of
        two numbers or as a keyword (if the surface is produced by one
        of the functions in ase.build).

    offset (default: None): Offsets the adsorbate by a number of unit
        cells. Mostly useful when adding more than one adsorbate.

    mol_index (default: 0): If the adsorbate is a molecule, index of
        the atom to be positioned above the location specified by the
        position argument.

    Note *position* is given in absolute xy coordinates (or as
    a keyword), whereas offset is specified in unit cells.  This
    can be used to give the positions in units of the unit cell by
    using *offset* instead.

    """
    info = slab.info.get('adsorbate_info', {})
    pos = np.array([0.0, 0.0])
    spos = np.array([0.0, 0.0])
    if offset is not None:
        spos += np.asarray(offset, float)
    if isinstance(position, str):
        if 'sites' not in info:
            raise TypeError('If the atoms are not made by an ' + 'ase.build function, ' + 'position cannot be a name.')
        if position not in info['sites']:
            raise TypeError('Adsorption site %s not supported.' % position)
        spos += info['sites'][position]
    else:
        pos += position
    if 'cell' in info:
        cell = info['cell']
    else:
        cell = slab.get_cell()[:2, :2]
    pos += np.dot(spos, cell)
    if isinstance(adsorbate, Atoms):
        ads = adsorbate
    elif isinstance(adsorbate, Atom):
        ads = Atoms([adsorbate])
    else:
        ads = Atoms([Atom(adsorbate)])
    if 'top layer atom index' in info:
        a = info['top layer atom index']
    else:
        a = slab.positions[:, 2].argmax()
        if 'adsorbate_info' not in slab.info:
            slab.info['adsorbate_info'] = {}
        slab.info['adsorbate_info']['top layer atom index'] = a
    z = slab.positions[a, 2] + height
    ads.translate([pos[0], pos[1], z] - ads.positions[mol_index])
    slab.extend(ads)
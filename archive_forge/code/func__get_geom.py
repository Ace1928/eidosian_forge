import os
import numpy as np
from copy import deepcopy
from ase.calculators.calculator import KPoints, kpts2kpts
def _get_geom(atoms, **params):
    geom_header = ['geometry units angstrom']
    for geomkw in ['center', 'autosym', 'autoz']:
        geom_header.append(geomkw if params.get(geomkw) else 'no' + geomkw)
    if 'geompar' in params:
        geom_header.append(params['geompar'])
    geom = [' '.join(geom_header)]
    outpos = atoms.get_positions()
    pbc = atoms.pbc
    if np.any(pbc):
        scpos = atoms.get_scaled_positions()
        for i, pbci in enumerate(pbc):
            if pbci:
                outpos[:, i] = scpos[:, i]
        npbc = pbc.sum()
        cellpars = atoms.cell.cellpar()
        geom.append('  system {} units angstrom'.format(_system_type[npbc]))
        if npbc == 3:
            geom.append('    lattice_vectors')
            for row in atoms.cell:
                geom.append('      {:20.16e} {:20.16e} {:20.16e}'.format(*row))
        else:
            if pbc[0]:
                geom.append('    lat_a {:20.16e}'.format(cellpars[0]))
            if pbc[1]:
                geom.append('    lat_b {:20.16e}'.format(cellpars[1]))
            if pbc[2]:
                geom.append('    lat_c {:20.16e}'.format(cellpars[2]))
            if pbc[1] and pbc[2]:
                geom.append('    alpha {:20.16e}'.format(cellpars[3]))
            if pbc[0] and pbc[2]:
                geom.append('    beta {:20.16e}'.format(cellpars[4]))
            if pbc[1] and pbc[0]:
                geom.append('    gamma {:20.16e}'.format(cellpars[5]))
        geom.append('  end')
    for i, atom in enumerate(atoms):
        geom.append('  {:<2} {:20.16e} {:20.16e} {:20.16e}'.format(atom.symbol, *outpos[i]))
    symm = params.get('symmetry')
    if symm is not None:
        geom.append('  symmetry {}'.format(symm))
    geom.append('end')
    return geom
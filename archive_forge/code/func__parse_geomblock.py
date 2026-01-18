import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
def _parse_geomblock(chunk):
    geomblocks = _geom.findall(chunk)
    if not geomblocks:
        return None
    geomblock = geomblocks[-1].strip().split('\n')
    natoms = len(geomblock)
    symbols = []
    pos = np.zeros((natoms, 3))
    for i, line in enumerate(geomblock):
        line = line.strip().split()
        symbols.append(line[1])
        pos[i] = [float(x) for x in line[3:6]]
    cellblocks = _cell_block.findall(chunk)
    if cellblocks:
        cellblock = cellblocks[-1].strip().split('\n')
        cell = np.zeros((3, 3))
        for i, line in enumerate(cellblock):
            line = line.strip().split()
            cell[i] = [float(x) for x in line[1:4]]
    else:
        cell = None
    return Atoms(symbols, positions=pos, cell=cell)
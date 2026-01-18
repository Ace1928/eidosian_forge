import re
import numpy as np
from ase import Atoms
from ase.utils import reader, writer
from ase.io.utils import ImageIterator
from ase.io import ParseError
from .vasp_parsers import vasp_outcar_parsers as vop
from pathlib import Path
def _write_xdatcar_config(fd, atoms, index):
    """Write a block of positions for XDATCAR file

    Args:
        fd (fd): writeable Python file descriptor
        atoms (ase.Atoms): Atoms to write
        index (int): configuration number written to block header

    """
    fd.write('Direct configuration={:6d}\n'.format(index))
    float_string = '{:11.8f}'
    scaled_positions = atoms.get_scaled_positions()
    for row in scaled_positions:
        fd.write(' ')
        fd.write(' '.join([float_string.format(x) for x in row]))
        fd.write('\n')
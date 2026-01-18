from __future__ import annotations
import os
import subprocess
import warnings
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.tempfile import ScratchDir
from pymatgen.core import Element
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar
@staticmethod
def _get_bond_order_info(filename):
    """Internal command to process pairwise bond order information.

        Args:
            filename (str): The path to the DDEC6_even_tempered_bond_orders.xyz file
        """
    bond_order_info = {}
    with open(filename) as r:
        for line in r:
            split = line.strip().split()
            if 'Printing BOs' in line:
                start_idx = int(split[5]) - 1
                start_el = Element(split[7])
                bond_order_info[start_idx] = {'element': start_el, 'bonded_to': []}
            elif 'Bonded to the' in line:
                direction = tuple((int(i.split(')')[0].split(',')[0]) for i in split[4:7]))
                end_idx = int(split[12]) - 1
                end_el = Element(split[14])
                bo = float(split[20])
                spin_bo = float(split[-1])
                bonded_to = {'index': end_idx, 'element': end_el, 'bond_order': bo, 'direction': direction, 'spin_polarization': spin_bo}
                bond_order_info[start_idx]['bonded_to'].append(bonded_to)
            elif 'The sum of bond orders for this atom' in line:
                bond_order_info[start_idx]['bond_order_sum'] = float(split[-1])
    return bond_order_info
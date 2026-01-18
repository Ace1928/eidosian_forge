from __future__ import annotations
import collections
import fnmatch
import os
import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import LobsterBandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import Dos, LobsterCompleteDos
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Vasprun, VolumetricData
from pymatgen.util.due import Doi, due
class Grosspop(MSONable):
    """
    Class to read in GROSSPOP.lobster files.

    Attributes:
        list_dict_grosspop (list[dict[str, str| dict[str, str]]]): List of dictionaries
            including all information about the grosspopulations. Each dictionary contains the following keys:
            - 'element': The element symbol of the atom.
            - 'Mulliken GP': A dictionary of Mulliken gross populations, where the keys are the orbital labels and the
                values are the corresponding gross populations as strings.
            - 'Loewdin GP': A dictionary of Loewdin gross populations, where the keys are the orbital labels and the
                values are the corresponding gross populations as strings.
            The 0th entry of the list refers to the first atom in GROSSPOP.lobster and so on.
    """

    def __init__(self, filename: str='GROSSPOP.lobster', list_dict_grosspop: list[dict] | None=None):
        """
        Args:
            filename: filename of the "GROSSPOP.lobster" file
            list_dict_grosspop: List of dictionaries including all information about the gross populations
        """
        self._filename = filename
        self.list_dict_grosspop = [] if list_dict_grosspop is None else list_dict_grosspop
        if not self.list_dict_grosspop:
            with zopen(filename, mode='rt') as file:
                contents = file.read().split('\n')
            small_dict: dict[str, Any] = {}
            for line in contents[3:]:
                cleanline = [i for i in line.split(' ') if i != '']
                if len(cleanline) == 5:
                    small_dict = {}
                    small_dict['Mulliken GP'] = {}
                    small_dict['Loewdin GP'] = {}
                    small_dict['element'] = cleanline[1]
                    small_dict['Mulliken GP'][cleanline[2]] = float(cleanline[3])
                    small_dict['Loewdin GP'][cleanline[2]] = float(cleanline[4])
                elif len(cleanline) > 0:
                    small_dict['Mulliken GP'][cleanline[0]] = float(cleanline[1])
                    small_dict['Loewdin GP'][cleanline[0]] = float(cleanline[2])
                    if 'total' in cleanline[0]:
                        self.list_dict_grosspop += [small_dict]

    def get_structure_with_total_grosspop(self, structure_filename: str) -> Structure:
        """
        Get a Structure with Mulliken and Loewdin total grosspopulations as site properties

        Args:
            structure_filename (str): filename of POSCAR

        Returns:
            Structure Object with Mulliken and Loewdin total grosspopulations as site properties.
        """
        struct = Structure.from_file(structure_filename)
        mullikengp = []
        loewdingp = []
        for grosspop in self.list_dict_grosspop:
            mullikengp += [grosspop['Mulliken GP']['total']]
            loewdingp += [grosspop['Loewdin GP']['total']]
        site_properties = {'Total Mulliken GP': mullikengp, 'Total Loewdin GP': loewdingp}
        return struct.copy(site_properties=site_properties)
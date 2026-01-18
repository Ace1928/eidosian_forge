from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.dev import requires
from monty.io import zopen
from monty.tempfile import ScratchDir
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.cssr import Cssr
from pymatgen.io.xyz import XYZ
class ZeoCssr(Cssr):
    """
    ZeoCssr adds extra fields to CSSR sites to conform with Zeo++
    input CSSR format. The coordinate system is rotated from xyz to zyx.
    This change aligns the pivot axis of pymatgen (z-axis) to pivot axis
    of Zeo++ (x-axis) for structural modifications.
    """

    def __init__(self, structure: Structure):
        """
        Args:
            structure: A structure to create ZeoCssr object.
        """
        super().__init__(structure)

    def __str__(self):
        """
        CSSR.__str__ method is modified to pad 0's to the CSSR site data.
        The padding is to conform with the CSSR format supported Zeo++.
        The oxidation state is stripped from site.specie
        Also coordinate system is rotated from xyz to zxy.
        """
        a, b, c = self.structure.lattice.lengths
        alpha, beta, gamma = self.structure.lattice.angles
        output = [f'{c:.4f} {a:.4f} {b:.4f}', f'{gamma:.2f} {alpha:.2f} {beta:.2f} SPGR =  1 P 1    OPT = 1', f'{len(self.structure)} 0', f'0 {self.structure.formula}']
        for idx, site in enumerate(self.structure):
            charge = getattr(site, 'charge', 0)
            specie = site.species_string
            output.append(f'{idx + 1} {specie} {site.c:.4f} {site.a:.4f} {site.b:.4f} 0 0 0 0 0 0 0 0 {charge:.4f}')
        return '\n'.join(output)

    @classmethod
    def from_str(cls, string: str) -> Self:
        """
        Reads a string representation to a ZeoCssr object.

        Args:
            string: A string representation of a ZeoCSSR.

        Returns:
            ZeoCssr object.
        """
        lines = string.split('\n')
        tokens = lines[0].split()
        lengths = [float(i) for i in tokens]
        tokens = lines[1].split()
        angles = [float(i) for i in tokens[0:3]]
        a = lengths.pop(-1)
        lengths.insert(0, a)
        alpha = angles.pop(-1)
        angles.insert(0, alpha)
        lattice = Lattice.from_parameters(*lengths, *angles)
        sp = []
        coords = []
        charge = []
        for line in lines[4:]:
            match = re.match('\\d+\\s+(\\w+)\\s+([0-9\\-\\.]+)\\s+([0-9\\-\\.]+)\\s+([0-9\\-\\.]+)\\s+(?:0\\s+){8}([0-9\\-\\.]+)', line.strip())
            if match:
                sp.append(match.group(1))
                coords.append([float(match.group(i)) for i in [3, 4, 2]])
                charge.append(match.group(5))
        return cls(Structure(lattice, sp, coords, site_properties={'charge': charge}))

    @classmethod
    def from_file(cls, filename: str | Path) -> Self:
        """
        Reads a CSSR file to a ZeoCssr object.

        Args:
            filename: Filename to read from.

        Returns:
            ZeoCssr object.
        """
        with zopen(filename, mode='r') as file:
            return cls.from_str(file.read())
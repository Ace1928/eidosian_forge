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
class ZeoVoronoiXYZ(XYZ):
    """
    Class to read Voronoi Nodes from XYZ file written by Zeo++.
    The sites have an additional column representing the voronoi node radius.
    The voronoi node radius is represented by the site property voronoi_radius.
    """

    def __init__(self, mol):
        """
        Args:
            mol: Input molecule holding the voronoi node information.
        """
        super().__init__(mol)

    @classmethod
    def from_str(cls, contents: str) -> Self:
        """
        Creates Zeo++ Voronoi XYZ object from a string.
        from_string method of XYZ class is being redefined.

        Args:
            contents: String representing Zeo++ Voronoi XYZ file.

        Returns:
            ZeoVoronoiXYZ object
        """
        lines = contents.split('\n')
        num_sites = int(lines[0])
        coords = []
        sp = []
        prop = []
        coord_patt = re.compile('(\\w+)\\s+([0-9\\-\\.]+)\\s+([0-9\\-\\.]+)\\s+([0-9\\-\\.]+)\\s+([0-9\\-\\.]+)')
        for i in range(2, 2 + num_sites):
            m = coord_patt.search(lines[i])
            if m:
                sp.append(m.group(1))
                coords.append([float(j) for j in [m.group(i) for i in [3, 4, 2]]])
                prop.append(float(m.group(5)))
        return cls(Molecule(sp, coords, site_properties={'voronoi_radius': prop}))

    @classmethod
    def from_file(cls, filename: str | Path) -> Self:
        """
        Creates XYZ object from a file.

        Args:
            filename: XYZ filename

        Returns:
            XYZ object
        """
        with zopen(filename) as file:
            return cls.from_str(file.read())

    def __str__(self) -> str:
        output = [str(len(self._mols[0])), self._mols[0].formula]
        prec = self.precision
        for site in self._mols[0]:
            x, y, z = site.coords
            symbol, voronoi_radius = (site.specie.symbol, site.properties['voronoi_radius'])
            output.append(f'{symbol} {z:.{prec}f} {x:.{prec}f} {y:.{prec}f} {voronoi_radius:.{prec}f}')
        return '\n'.join(output)
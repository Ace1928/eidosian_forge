from __future__ import annotations
import re
from collections import defaultdict
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.re import regrep
from pymatgen.core import Element, Lattice, Structure
from pymatgen.util.io_utils import clean_lines
class PWOutput:
    """Parser for PWSCF output file."""
    patterns = dict(energies='total energy\\s+=\\s+([\\d\\.\\-]+)\\sRy', ecut='kinetic\\-energy cutoff\\s+=\\s+([\\d\\.\\-]+)\\s+Ry', lattice_type='bravais\\-lattice index\\s+=\\s+(\\d+)', celldm1='celldm\\(1\\)=\\s+([\\d\\.]+)\\s', celldm2='celldm\\(2\\)=\\s+([\\d\\.]+)\\s', celldm3='celldm\\(3\\)=\\s+([\\d\\.]+)\\s', celldm4='celldm\\(4\\)=\\s+([\\d\\.]+)\\s', celldm5='celldm\\(5\\)=\\s+([\\d\\.]+)\\s', celldm6='celldm\\(6\\)=\\s+([\\d\\.]+)\\s', nkpts='number of k points=\\s+([\\d]+)')

    def __init__(self, filename):
        """
        Args:
            filename (str): Filename.
        """
        self.filename = filename
        self.data = defaultdict(list)
        self.read_pattern(PWOutput.patterns)
        for k, v in self.data.items():
            if k == 'energies':
                self.data[k] = [float(i[0][0]) for i in v]
            elif k in ['lattice_type', 'nkpts']:
                self.data[k] = int(v[0][0][0])
            else:
                self.data[k] = float(v[0][0][0])

    def read_pattern(self, patterns, reverse=False, terminate_on_match=False, postprocess=str):
        """
        General pattern reading. Uses monty's regrep method. Takes the same
        arguments.

        Args:
            patterns (dict): A dict of patterns, e.g.,
                {"energy": r"energy\\\\(sigma->0\\\\)\\\\s+=\\\\s+([\\\\d\\\\-.]+)"}.
            reverse (bool): Read files in reverse. Defaults to false. Useful for
                large files, esp OUTCARs, especially when used with
                terminate_on_match.
            terminate_on_match (bool): Whether to terminate when there is at
                least one match in each key in pattern.
            postprocess (callable): A post processing function to convert all
                matches. Defaults to str, i.e., no change.

        Renders accessible:
            Any attribute in patterns. For example,
            {"energy": r"energy\\\\(sigma->0\\\\)\\\\s+=\\\\s+([\\\\d\\\\-.]+)"} will set the
            value of self.data["energy"] = [[-1234], [-3453], ...], to the
            results from regex and postprocess. Note that the returned
            values are lists of lists, because you can grep multiple
            items on one line.
        """
        matches = regrep(self.filename, patterns, reverse=reverse, terminate_on_match=terminate_on_match, postprocess=postprocess)
        self.data.update(matches)

    def get_celldm(self, idx: int):
        """
        Args:
            idx (int): index.

        Returns:
            Cell dimension along index
        """
        return self.data[f'celldm{idx}']

    @property
    def final_energy(self):
        """Returns: Final energy."""
        return self.data['energies'][-1]

    @property
    def lattice_type(self):
        """Returns: Lattice type."""
        return self.data['lattice_type']
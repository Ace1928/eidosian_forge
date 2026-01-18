from __future__ import annotations
import os
import re
import warnings
from string import Template
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.analysis.excitation import ExcitationSpectrum
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.units import Energy, FloatWithUnit
class NwInput(MSONable):
    """
    An object representing a Nwchem input file, which is essentially a list
    of tasks on a particular molecule.
    """

    def __init__(self, mol, tasks, directives=None, geometry_options=('units', 'angstroms'), symmetry_options=None, memory_options=None):
        """
        Args:
            mol: Input molecule. If molecule is a single string, it is used as a
                direct input to the geometry section of the Gaussian input
                file.
            tasks: List of NwTasks.
            directives: List of root level directives as tuple. E.g.,
                [("start", "water"), ("print", "high")]
            geometry_options: Additional list of options to be supplied to the
                geometry. E.g., ["units", "angstroms", "noautoz"]. Defaults to
                ("units", "angstroms").
            symmetry_options: Addition list of option to be supplied to the
                symmetry. E.g. ["c1"] to turn off the symmetry
            memory_options: Memory controlling options. str.
                E.g "total 1000 mb stack 400 mb".
        """
        self._mol = mol
        self.directives = directives if directives is not None else []
        self.tasks = tasks
        self.geometry_options = geometry_options
        self.symmetry_options = symmetry_options
        self.memory_options = memory_options

    @property
    def molecule(self):
        """Returns molecule associated with this GaussianInput."""
        return self._mol

    def __str__(self):
        out = []
        if self.memory_options:
            out.append('memory ' + self.memory_options)
        for d in self.directives:
            out.append(f'{d[0]} {d[1]}')
        out.append('geometry ' + ' '.join(self.geometry_options))
        if self.symmetry_options:
            out.append(' symmetry ' + ' '.join(self.symmetry_options))
        for site in self._mol:
            out.append(f' {site.specie.symbol} {site.x} {site.y} {site.z}')
        out.append('end\n')
        for task in self.tasks:
            out.extend((str(task), ''))
        return '\n'.join(out)

    def write_file(self, filename):
        """
        Args:
            filename (str): Filename.
        """
        with zopen(filename, mode='w') as file:
            file.write(str(self))

    def as_dict(self):
        """Returns: MSONable dict."""
        return {'mol': self._mol.as_dict(), 'tasks': [task.as_dict() for task in self.tasks], 'directives': [list(task) for task in self.directives], 'geometry_options': list(self.geometry_options), 'symmetry_options': self.symmetry_options, 'memory_options': self.memory_options}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            NwInput
        """
        return cls(Molecule.from_dict(dct['mol']), tasks=[NwTask.from_dict(dt) for dt in dct['tasks']], directives=[tuple(li) for li in dct['directives']], geometry_options=dct['geometry_options'], symmetry_options=dct['symmetry_options'], memory_options=dct['memory_options'])

    @classmethod
    def from_str(cls, string_input: str) -> Self:
        """
        Read an NwInput from a string. Currently tested to work with
        files generated from this class itself.

        Args:
            string_input: string_input to parse.

        Returns:
            NwInput object
        """
        directives = []
        tasks = []
        charge = spin_multiplicity = title = basis_set = None
        basis_set_option = None
        theory_directives: dict[str, dict[str, str]] = {}
        geom_options = symmetry_options = memory_options = None
        lines = string_input.strip().split('\n')
        while len(lines) > 0:
            line = lines.pop(0).strip()
            if line == '':
                continue
            tokens = line.split()
            if tokens[0].lower() == 'geometry':
                geom_options = tokens[1:]
                line = lines.pop(0).strip()
                tokens = line.split()
                if tokens[0].lower() == 'symmetry':
                    symmetry_options = tokens[1:]
                    line = lines.pop(0).strip()
                species = []
                coords = []
                while line.lower() != 'end':
                    tokens = line.split()
                    species.append(tokens[0])
                    coords.append([float(i) for i in tokens[1:]])
                    line = lines.pop(0).strip()
                mol = Molecule(species, coords)
            elif tokens[0].lower() == 'charge':
                charge = int(tokens[1])
            elif tokens[0].lower() == 'title':
                title = line[5:].strip().strip('"')
            elif tokens[0].lower() == 'basis':
                line = lines.pop(0).strip()
                basis_set = {}
                while line.lower() != 'end':
                    tokens = line.split()
                    basis_set[tokens[0]] = tokens[-1].strip('"')
                    line = lines.pop(0).strip()
            elif tokens[0].lower() in NwTask.theories:
                if len(tokens) > 1:
                    basis_set_option = tokens[1]
                theory = tokens[0].lower()
                line = lines.pop(0).strip()
                theory_directives[theory] = {}
                while line.lower() != 'end':
                    tokens = line.split()
                    theory_directives[theory][tokens[0]] = tokens[-1]
                    if tokens[0] == 'mult':
                        spin_multiplicity = float(tokens[1])
                    line = lines.pop(0).strip()
            elif tokens[0].lower() == 'task':
                tasks.append(NwTask(charge=charge, spin_multiplicity=spin_multiplicity, title=title, theory=tokens[1], operation=tokens[2], basis_set=basis_set, basis_set_option=basis_set_option, theory_directives=theory_directives.get(tokens[1])))
            elif tokens[0].lower() == 'memory':
                memory_options = ' '.join(tokens[1:])
            else:
                directives.append(line.strip().split())
        return cls(mol, tasks=tasks, directives=directives, geometry_options=geom_options, symmetry_options=symmetry_options, memory_options=memory_options)

    @classmethod
    def from_file(cls, filename: str | Path) -> Self:
        """
        Read an NwInput from a file. Currently tested to work with
        files generated from this class itself.

        Args:
            filename: Filename to parse.

        Returns:
            NwInput object
        """
        with zopen(filename) as file:
            return cls.from_str(file.read())
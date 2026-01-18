from __future__ import annotations
import abc
import json
import logging
import os
import warnings
from glob import glob
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MSONable
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.gaussian import GaussianOutput
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar
from pymatgen.io.vasp.outputs import Dynmat, Oszicar, Vasprun
class SimpleVaspToComputedEntryDrone(VaspToComputedEntryDrone):
    """A simpler VaspToComputedEntryDrone. Instead of parsing vasprun.xml, it
    parses only the INCAR, POTCAR, OSZICAR and KPOINTS files, which are much
    smaller and faster to parse. However, much fewer properties are available
    compared to the standard VaspToComputedEntryDrone.
    """

    def __init__(self, inc_structure=False):
        """
        Args:
            inc_structure (bool): Set to True if you want ComputedStructureEntries to be returned instead of
                ComputedEntries. Structure will be parsed from the CONTCAR.
        """
        self._inc_structure = inc_structure
        self._parameters = {'is_hubbard', 'hubbards', 'potcar_spec', 'run_type'}

    def assimilate(self, path):
        """Assimilate data in a directory path into a ComputedEntry object.

        Args:
            path: directory path

        Returns:
            ComputedEntry
        """
        files = os.listdir(path)
        try:
            files_to_parse = {}
            filenames = {'INCAR', 'POTCAR', 'CONTCAR', 'OSZICAR', 'POSCAR', 'DYNMAT'}
            if 'relax1' in files and 'relax2' in files:
                for filename in ('INCAR', 'POTCAR', 'POSCAR'):
                    search_str = (f'{path}/relax1', filename + '*')
                    files_to_parse[filename] = glob(search_str)[0]
                for filename in ('CONTCAR', 'OSZICAR'):
                    search_str = (f'{path}/relax2', filename + '*')
                    files_to_parse[filename] = glob(search_str)[-1]
            else:
                for filename in filenames:
                    files = sorted(glob(os.path.join(path, filename + '*')))
                    if len(files) == 1 or filename in ('INCAR', 'POTCAR') or (len(files) == 1 and filename == 'DYNMAT'):
                        files_to_parse[filename] = files[0]
                    elif len(files) > 1:
                        files_to_parse[filename] = files[0] if filename == 'POSCAR' else files[-1]
                        warnings.warn(f'{len(files)} files found. {files_to_parse[filename]} is being parsed.')
            if not set(files_to_parse).issuperset({'INCAR', 'POTCAR', 'CONTCAR', 'OSZICAR', 'POSCAR'}):
                raise ValueError(f'Unable to parse {files_to_parse} as not all necessary files are present! SimpleVaspToComputedEntryDrone requires INCAR, POTCAR, CONTCAR, OSZICAR, POSCAR to be present. Only {files} detected')
            poscar = Poscar.from_file(files_to_parse['POSCAR'])
            contcar = Poscar.from_file(files_to_parse['CONTCAR'])
            incar = Incar.from_file(files_to_parse['INCAR'])
            potcar = Potcar.from_file(files_to_parse['POTCAR'])
            oszicar = Oszicar(files_to_parse['OSZICAR'])
            param = {'hubbards': {}}
            if 'LDAUU' in incar:
                param['hubbards'] = dict(zip(poscar.site_symbols, incar['LDAUU']))
            param['is_hubbard'] = incar.get('LDAU', True) and sum(param['hubbards'].values()) > 0
            param['run_type'] = None
            param['potcar_spec'] = potcar.spec
            energy = oszicar.final_energy
            structure = contcar.structure
            initial_vol = poscar.structure.volume
            final_vol = contcar.structure.volume
            delta_volume = final_vol / initial_vol - 1
            data = {'filename': path, 'delta_volume': delta_volume}
            if 'DYNMAT' in files_to_parse:
                dynmat = Dynmat(files_to_parse['DYNMAT'])
                data['phonon_frequencies'] = dynmat.get_phonon_frequencies()
            if self._inc_structure:
                return ComputedStructureEntry(structure, energy, parameters=param, data=data)
            return ComputedEntry(structure.composition, energy, parameters=param, data=data)
        except Exception as exc:
            logger.debug(f'error in {path}: {exc}')
            return None

    def __str__(self):
        return 'SimpleVaspToComputedEntryDrone'

    def as_dict(self):
        """Returns: MSONable dict."""
        return {'init_args': {'inc_structure': self._inc_structure}, '@module': type(self).__module__, '@class': type(self).__name__}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict Representation.

        Returns:
            SimpleVaspToComputedEntryDrone
        """
        return cls(**dct['init_args'])
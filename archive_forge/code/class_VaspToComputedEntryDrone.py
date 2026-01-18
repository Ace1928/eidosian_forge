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
class VaspToComputedEntryDrone(AbstractDrone):
    """VaspToEntryDrone assimilates directories containing VASP output to
    ComputedEntry/ComputedStructureEntry objects.

    There are some restrictions on the valid directory structures:

    1. There can be only one vasp run in each directory.
    2. Directories designated "relax1", "relax2" are considered to be 2 parts
       of an aflow style run, and only "relax2" is parsed.
    3. The drone parses only the vasprun.xml file.
    """

    def __init__(self, inc_structure=False, parameters=None, data=None):
        """
        Args:
            inc_structure (bool): Set to True if you want
                ComputedStructureEntries to be returned instead of
                ComputedEntries.
            parameters (list): Input parameters to include. It has to be one of
                the properties supported by the Vasprun object. See
                pymatgen.io.vasp.Vasprun. If parameters is None,
                a default set of parameters that are necessary for typical
                post-processing will be set.
            data (list): Output data to include. Has to be one of the properties
                supported by the Vasprun object.
        """
        self._inc_structure = inc_structure
        self._parameters = {'is_hubbard', 'hubbards', 'potcar_spec', 'potcar_symbols', 'run_type'}
        if parameters:
            self._parameters.update(parameters)
        self._data = data or []

    def assimilate(self, path):
        """Assimilate data in a directory path into a ComputedEntry object.

        Args:
            path: directory path

        Returns:
            ComputedEntry
        """
        files = os.listdir(path)
        if 'relax1' in files and 'relax2' in files:
            filepath = glob(f'{path}/relax2/vasprun.xml*')[0]
        else:
            vasprun_files = glob(f'{path}/vasprun.xml*')
            filepath = None
            if len(vasprun_files) == 1:
                filepath = vasprun_files[0]
            elif len(vasprun_files) > 1:
                filepath = sorted(vasprun_files)[-1]
                warnings.warn(f'{len(vasprun_files)} vasprun.xml.* found. {filepath} is being parsed.')
        try:
            vasp_run = Vasprun(filepath)
        except Exception as exc:
            logger.debug(f'error in {filepath}: {exc}')
            return None
        return vasp_run.get_computed_entry(self._inc_structure, parameters=self._parameters, data=self._data)

    def get_valid_paths(self, path):
        """Checks if paths contains vasprun.xml or (POSCAR+OSZICAR).

        Args:
            path: input path as a tuple generated from os.walk, i.e.,
                (parent, subdirs, files).

        Returns:
            List of valid dir/file paths for assimilation
        """
        parent, subdirs, _files = path
        if 'relax1' in subdirs and 'relax2' in subdirs:
            return [parent]
        if not parent.endswith('/relax1') and (not parent.endswith('/relax2')) and (len(glob(f'{parent}/vasprun.xml*')) > 0 or (len(glob(f'{parent}/POSCAR*')) > 0 and len(glob(f'{parent}/OSZICAR*')) > 0)):
            return [parent]
        return []

    def __str__(self):
        return ' VaspToComputedEntryDrone'

    def as_dict(self):
        """Returns: MSONABle dict."""
        return {'init_args': {'inc_structure': self._inc_structure, 'parameters': self._parameters, 'data': self._data}, '@module': type(self).__module__, '@class': type(self).__name__}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict Representation.

        Returns:
            VaspToComputedEntryDrone
        """
        return cls(**dct['init_args'])
from __future__ import annotations
import abc
import itertools
import os
import re
import shutil
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast
from zipfile import ZipFile
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, PeriodicSite, SiteCollection, Species, Structure
from pymatgen.io.core import InputGenerator
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due
class MITNEBSet(DictSet):
    """
    Class for writing NEB inputs.

    Note that EDIFF is not on a per atom basis for this input set.
    """

    def __init__(self, structures, unset_encut=False, **kwargs):
        """
        Args:
            structures: List of Structure objects.
            unset_encut (bool): Whether to unset ENCUT.
            **kwargs: Other kwargs supported by DictSet.
        """
        if len(structures) < 3:
            raise ValueError(f'You need at least 3 structures for an NEB, got {len(structures)}')
        kwargs['sort_structure'] = False
        super().__init__(structures[0], MITRelaxSet.CONFIG, **kwargs)
        self.structures = self._process_structures(structures)
        self.unset_encut = False
        if unset_encut:
            self._config_dict['INCAR'].pop('ENCUT', None)
        if 'EDIFF' not in self._config_dict['INCAR']:
            self._config_dict['INCAR']['EDIFF'] = self._config_dict['INCAR'].pop('EDIFF_PER_ATOM')
        defaults = {'IMAGES': len(structures) - 2, 'IBRION': 1, 'ISYM': 0, 'LCHARG': False, 'LDAU': False}
        self._config_dict['INCAR'].update(defaults)

    @property
    def poscar(self):
        """Poscar for structure of first end point."""
        return Poscar(self.structures[0])

    @property
    def poscars(self):
        """List of Poscars."""
        return [Poscar(s) for s in self.structures]

    @staticmethod
    def _process_structures(structures):
        """Remove any atom jumps across the cell."""
        input_structures = structures
        structures = [input_structures[0]]
        for s in input_structures[1:]:
            prev = structures[-1]
            for idx, site in enumerate(s):
                translate = np.round(prev[idx].frac_coords - site.frac_coords)
                if np.any(np.abs(translate) > 0.5):
                    s.translate_sites([idx], translate, to_unit_cell=False)
            structures.append(s)
        return structures

    def write_input(self, output_dir, make_dir_if_not_present=True, write_cif=False, write_path_cif=False, write_endpoint_inputs=False):
        """
        NEB inputs has a special directory structure where inputs are in 00,
        01, 02, ....

        Args:
            output_dir (str): Directory to output the VASP input files
            make_dir_if_not_present (bool): Set to True if you want the
                directory (and the whole path) to be created if it is not
                present.
            write_cif (bool): If true, writes a cif along with each POSCAR.
            write_path_cif (bool): If true, writes a cif for each image.
            write_endpoint_inputs (bool): If true, writes input files for
                running endpoint calculations.
        """
        output_dir = Path(output_dir)
        if make_dir_if_not_present and (not output_dir.exists()):
            output_dir.mkdir(parents=True)
        self.incar.write_file(str(output_dir / 'INCAR'))
        self.kpoints.write_file(str(output_dir / 'KPOINTS'))
        self.potcar.write_file(str(output_dir / 'POTCAR'))
        for idx, poscar in enumerate(self.poscars):
            d = output_dir / str(idx).zfill(2)
            if not d.exists():
                d.mkdir(parents=True)
            poscar.write_file(str(d / 'POSCAR'))
            if write_cif:
                poscar.structure.to(filename=str(d / f'{idx}.cif'))
        if write_endpoint_inputs:
            end_point_param = MITRelaxSet(self.structures[0], user_incar_settings=self.user_incar_settings)
            for image in ['00', str(len(self.structures) - 1).zfill(2)]:
                end_point_param.incar.write_file(str(output_dir / image / 'INCAR'))
                end_point_param.kpoints.write_file(str(output_dir / image / 'KPOINTS'))
                end_point_param.potcar.write_file(str(output_dir / image / 'POTCAR'))
        if write_path_cif:
            sites = {PeriodicSite(site.species, site.frac_coords, self.structures[0].lattice) for site in chain(*(struct for struct in self.structures))}
            neb_path = Structure.from_sites(sorted(sites))
            neb_path.to(filename=f'{output_dir}/path.cif')
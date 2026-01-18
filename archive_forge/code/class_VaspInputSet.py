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
class VaspInputSet(InputGenerator, abc.ABC):
    """
    Base class representing a set of VASP input parameters with a structure
    supplied as init parameters. Typically, you should not inherit from this
    class. Start from DictSet or MPRelaxSet or MITRelaxSet.
    """
    _valid_potcars: Sequence[str] | None = None

    @property
    @abc.abstractmethod
    def incar(self):
        """Incar object."""

    @property
    @abc.abstractmethod
    def kpoints(self):
        """Kpoints object."""

    @property
    @abc.abstractmethod
    def poscar(self):
        """Poscar object."""

    @property
    def potcar_symbols(self):
        """List of POTCAR symbols."""
        elements = self.poscar.site_symbols
        potcar_symbols = []
        settings = self._config_dict['POTCAR']
        if isinstance(settings[elements[-1]], dict):
            for el in elements:
                potcar_symbols.append(settings[el]['symbol'] if el in settings else el)
        else:
            for el in elements:
                potcar_symbols.append(settings.get(el, el))
        return potcar_symbols

    @property
    def potcar(self) -> Potcar:
        """Potcar object."""
        user_potcar_functional = self.user_potcar_functional
        potcar = Potcar(self.potcar_symbols, functional=user_potcar_functional)
        for p_single in potcar:
            if user_potcar_functional not in p_single.identify_potcar()[0]:
                warnings.warn(f'POTCAR data with symbol {p_single.symbol} is not known by pymatgen to correspond with the selected user_potcar_functional={user_potcar_functional!r}. This POTCAR is known to correspond with functionals {p_single.identify_potcar(mode='data')[0]}. Please verify that you are using the right POTCARs!', BadInputSetWarning)
        return potcar

    @deprecated(message='get_vasp_input will be removed in a future version of pymatgen. Use get_input_set instead.')
    def get_vasp_input(self, structure=None) -> VaspInput:
        """Get a VaspInput object.

        Returns:
            VaspInput.
        """
        return self.get_input_set(structure=structure)

    def get_input_set(self, structure=None) -> VaspInput:
        """Get a VaspInput object.

        Returns:
            VaspInput.
        """
        if structure is not None:
            self.structure = structure
        return VaspInput(incar=self.incar, kpoints=self.kpoints, poscar=self.poscar, potcar=self.potcar)

    def write_input(self, output_dir: str, make_dir_if_not_present: bool=True, include_cif: bool=False, potcar_spec: bool=False, zip_output: bool=False) -> None:
        """
        Writes a set of VASP input to a directory.

        Args:
            output_dir (str): Directory to output the VASP input files
            make_dir_if_not_present (bool): Set to True if you want the
                directory (and the whole path) to be created if it is not
                present.
            include_cif (bool): Whether to write a CIF file in the output
                directory for easier opening by VESTA.
            potcar_spec (bool): Instead of writing the POTCAR, write a "POTCAR.spec".
                This is intended to help sharing an input set with people who might
                not have a license to specific Potcar files. Given a "POTCAR.spec",
                the specific POTCAR file can be re-generated using pymatgen with the
                "generate_potcar" function in the pymatgen CLI.
            zip_output (bool): If True, output will be zipped into a file with the
                same name as the InputSet (e.g., MPStaticSet.zip)
        """
        if potcar_spec:
            if make_dir_if_not_present:
                os.makedirs(output_dir, exist_ok=True)
            with zopen(f'{output_dir}/POTCAR.spec', mode='wt') as file:
                file.write('\n'.join(self.potcar_symbols))
            for key in ['INCAR', 'POSCAR', 'KPOINTS']:
                if (val := getattr(self, key.lower())) is not None:
                    with zopen(os.path.join(output_dir, key), mode='wt') as file:
                        file.write(str(val))
        else:
            vasp_input = self.get_input_set()
            vasp_input.write_input(output_dir, make_dir_if_not_present=make_dir_if_not_present)
        cif_name = ''
        if include_cif:
            struct = vasp_input['POSCAR'].structure
            cif_name = f'{output_dir}/{struct.formula.replace(' ', '')}.cif'
            struct.to(filename=cif_name)
        if zip_output:
            filename = type(self).__name__ + '.zip'
            with ZipFile(os.path.join(output_dir, filename), mode='w') as zip_file:
                for file in ['INCAR', 'POSCAR', 'KPOINTS', 'POTCAR', 'POTCAR.spec', cif_name]:
                    try:
                        zip_file.write(os.path.join(output_dir, file), arcname=file)
                    except FileNotFoundError:
                        pass
                    try:
                        os.remove(os.path.join(output_dir, file))
                    except (FileNotFoundError, PermissionError, IsADirectoryError):
                        pass

    def as_dict(self, verbosity=2):
        """
        Args:
            verbosity: Verbosity for generated dict. If 1, structure is
            excluded.

        Returns:
            MSONable dict
        """
        dct = MSONable.as_dict(self)
        if verbosity == 1:
            dct.pop('structure', None)
        return dct
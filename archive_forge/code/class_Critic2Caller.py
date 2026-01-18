from __future__ import annotations
import logging
import os
import subprocess
import warnings
from enum import Enum, unique
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from scipy.spatial import KDTree
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import DummySpecies
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar, VolumetricData
from pymatgen.util.due import Doi, due
class Critic2Caller:
    """Class to call critic2 and store standard output for further processing."""

    @requires(which('critic2'), 'Critic2Caller requires the executable critic to be in the path. Please follow the instructions at https://github.com/aoterodelaroza/critic2.')
    def __init__(self, input_script: str):
        """Run Critic2 on a given input script.

        Args:
            input_script: string defining the critic2 input
        """
        self._input_script = input_script
        with open('input_script.cri', mode='w', encoding='utf-8') as file:
            file.write(input_script)
        args = ['critic2', 'input_script.cri']
        with subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE, close_fds=True) as rs:
            _stdout, _stderr = rs.communicate()
        stdout = _stdout.decode()
        if _stderr:
            stderr = _stderr.decode()
            warnings.warn(stderr)
        if rs.returncode != 0:
            raise RuntimeError(f'critic2 exited with return code {rs.returncode}: {stdout}')
        self._stdout = stdout
        self._stderr = stderr
        cp_report = loadfn('cpreport.json') if os.path.isfile('cpreport.json') else None
        self._cp_report = cp_report
        yt = loadfn('yt.json') if os.path.isfile('yt.json') else None
        self._yt = yt

    @classmethod
    def from_chgcar(cls, structure, chgcar=None, chgcar_ref=None, user_input_settings=None, write_cml=False, write_json=True, zpsp=None) -> Self:
        """Run Critic2 in automatic mode on a supplied structure, charge
        density (chgcar) and reference charge density (chgcar_ref).

        The reason for a separate reference field is that in
        VASP, the CHGCAR charge density only contains valence
        electrons and may be missing substantial charge at
        nuclei leading to misleading results. Thus, a reference
        field is commonly constructed from the sum of AECCAR0
        and AECCAR2 which is the total charge density, but then
        the valence charge density is used for the final analysis.

        If chgcar_ref is not supplied, chgcar will be used as the
        reference field. If chgcar is not supplied, the promolecular
        charge density will be used as the reference field -- this can
        often still give useful results if only topological information
        is wanted.

        User settings is a dictionary that can contain:
        * GRADEPS, float (field units), gradient norm threshold
        * CPEPS, float (Bohr units in crystals), minimum distance between
          critical points for them to be equivalent
        * NUCEPS, same as CPEPS but specifically for nucleus critical
          points (critic2 default is dependent on grid dimensions)
        * NUCEPSH, same as NUCEPS but specifically for hydrogen nuclei
          since associated charge density can be significantly displaced
          from hydrogen nucleus
        * EPSDEGEN, float (field units), discard critical point if any
          element of the diagonal of the Hessian is below this value,
          useful for discarding points in vacuum regions
        * DISCARD, float (field units), discard critical points with field
          value below this value, useful for discarding points in vacuum
          regions
        * SEED, list of strings, strategies for seeding points, default
          is ['WS 1', 'PAIR 10'] which seeds critical points by
          sub-dividing the Wigner-Seitz cell and between every atom pair
          closer than 10 Bohr, see critic2 manual for more options

        Args:
            structure: Structure to analyze
            chgcar: Charge density to use for analysis. If None, will
                use promolecular density. Should be a Chgcar object or path (string).
            chgcar_ref: Reference charge density. If None, will use
                chgcar as reference. Should be a Chgcar object or path (string).
            user_input_settings (dict): as explained above
            write_cml (bool): Useful for debug, if True will write all
                critical points to a file 'table.cml' in the working directory
                useful for visualization
            write_json (bool): Whether to write out critical points
                and YT json. YT integration will be performed with this setting.
            zpsp (dict): Dict of element/symbol name to number of electrons
                (ZVAL in VASP pseudopotential), with which to properly augment core regions
                and calculate charge transfer. Optional.
        """
        settings = {'CPEPS': 0.1, 'SEED': ['WS', 'PAIR DIST 10']}
        if user_input_settings:
            settings.update(user_input_settings)
        input_script = ['crystal POSCAR']
        if chgcar_ref:
            input_script += ['load ref.CHGCAR id chg_ref', 'reference chg_ref']
        if chgcar:
            input_script += ['load int.CHGCAR id chg_int', 'integrable chg_int']
            if zpsp:
                zpsp_str = f' zpsp {' '.join((f'{symbol} {int(zval)}' for symbol, zval in zpsp.items()))}'
                input_script[-2] += zpsp_str
        auto = 'auto '
        for k, v in settings.items():
            if isinstance(v, list):
                for item in v:
                    auto += f'{k} {item} '
            else:
                auto += f'{k} {v} '
        input_script += [auto]
        if write_cml:
            input_script += ['cpreport ../table.cml cell border graph']
        if write_json:
            input_script += ['cpreport cpreport.json']
        if write_json and chgcar:
            input_script += ['yt']
            input_script += ['yt JSON yt.json']
        input_script_str = '\n'.join(input_script)
        with ScratchDir('.'):
            structure.to(filename='POSCAR')
            if chgcar and isinstance(chgcar, VolumetricData):
                chgcar.write_file('int.CHGCAR')
            elif chgcar:
                os.symlink(chgcar, 'int.CHGCAR')
            if chgcar_ref and isinstance(chgcar_ref, VolumetricData):
                chgcar_ref.write_file('ref.CHGCAR')
            elif chgcar_ref:
                os.symlink(chgcar_ref, 'ref.CHGCAR')
            caller = cls(input_script_str)
            caller.output = Critic2Analysis(structure, stdout=caller._stdout, stderr=caller._stderr, cpreport=caller._cp_report, yt=caller._yt, zpsp=zpsp)
            return caller

    @classmethod
    def from_path(cls, path, suffix='', zpsp=None) -> Self:
        """Convenience method to run critic2 analysis on a folder with typical VASP output files.

        This method will:

        1. Look for files CHGCAR, AECAR0, AECAR2, POTCAR or their gzipped
        counterparts.

        2. If AECCAR* files are present, constructs a temporary reference
        file as AECCAR0 + AECCAR2.

        3. Runs critic2 analysis twice: once for charge, and a second time
        for the charge difference (magnetization density).

        Args:
            path: path to folder to search in
            suffix: specific suffix to look for (e.g. '.relax1' for
                'CHGCAR.relax1.gz')
            zpsp: manually specify ZPSP if POTCAR not present
        """
        chgcar_path = get_filepath('CHGCAR', 'Could not find CHGCAR!', path, suffix)
        chgcar = Chgcar.from_file(chgcar_path)
        chgcar_ref = None
        if not zpsp:
            potcar_path = get_filepath('POTCAR', 'Could not find POTCAR, will not be able to calculate charge transfer.', path, suffix)
            if potcar_path:
                potcar = Potcar.from_file(potcar_path)
                zpsp = {p.element: p.zval for p in potcar}
        if not zpsp:
            aeccar0_path = get_filepath('AECCAR0', 'Could not find AECCAR0, interpret Bader results with caution.', path, suffix)
            aeccar0 = Chgcar.from_file(aeccar0_path) if aeccar0_path else None
            aeccar2_path = get_filepath('AECCAR2', 'Could not find AECCAR2, interpret Bader results with caution.', path, suffix)
            aeccar2 = Chgcar.from_file(aeccar2_path) if aeccar2_path else None
            chgcar_ref = aeccar0.linear_add(aeccar2) if aeccar0 and aeccar2 else None
        return cls.from_chgcar(chgcar.structure, chgcar, chgcar_ref, zpsp=zpsp)
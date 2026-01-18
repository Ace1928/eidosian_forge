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
@dataclass
class DictSet(VaspInputSet):
    """
    Concrete implementation of VaspInputSet that is initialized from a dict
    settings. This allows arbitrary settings to be input. In general,
    this is rarely used directly unless there is a source of settings in yaml
    format (e.g., from a REST interface). It is typically used by other
    VaspInputSets for initialization.

    Special consideration should be paid to the way the MAGMOM initialization
    for the INCAR is done. The initialization differs depending on the type of
    structure and the configuration settings. The order in which the magmom is
    determined is as follows:

    1. If the site itself has a magmom setting (i.e. site.properties["magmom"] = float),
        that is used. This can be set with structure.add_site_property().
    2. If the species of the site has a spin setting, that is used. This can be set
        with structure.add_spin_by_element().
    3. If the species itself has a particular setting in the config file, that
       is used, e.g., Mn3+ may have a different magmom than Mn4+.
    4. Lastly, the element symbol itself is checked in the config file. If
       there are no settings, a default value of 0.6 is used.

    Args:
        structure (Structure): The Structure to create inputs for. If None, the input
            set is initialized without a Structure but one must be set separately before
            the inputs are generated.
        config_dict (dict): The config dictionary to use.
        files_to_transfer (dict): A dictionary of {filename: filepath}. This allows the
            transfer of files from a previous calculation.
        user_incar_settings (dict): User INCAR settings. This allows a user to override
            INCAR settings, e.g., setting a different MAGMOM for various elements or
            species. Note that in the new scheme, ediff_per_atom and hubbard_u are no
            longer args. Instead, the CONFIG supports EDIFF_PER_ATOM and EDIFF keys.
            The former scales with # of atoms, the latter does not. If both are present,
            EDIFF is preferred. To force such settings, just supply
            user_incar_settings={"EDIFF": 1e-5, "LDAU": False} for example. The keys
            'LDAUU', 'LDAUJ', 'LDAUL' are special cases since pymatgen defines different
            values depending on what anions are present in the structure, so these keys
            can be defined in one of two ways, e.g. either {"LDAUU":{"O":{"Fe":5}}} to
            set LDAUU for Fe to 5 in an oxide, or {"LDAUU":{"Fe":5}} to set LDAUU to 5
            regardless of the input structure. If a None value is given, that key is
            unset. For example, {"ENCUT": None} will remove ENCUT from the
            incar settings. Finally, KSPACING is a special setting and can be set to
            "auto" in which the KSPACING is set automatically based on the band gap.
        user_kpoints_settings (dict or Kpoints): Allow user to override kpoints setting
            by supplying a dict. E.g., {"reciprocal_density": 1000}. User can also
            supply Kpoints object.
        user_potcar_settings (dict): Allow user to override POTCARs. E.g., {"Gd":
            "Gd_3"}. This is generally not recommended.
        constrain_total_magmom (bool): Whether to constrain the total magmom (NUPDOWN in
            INCAR) to be the sum of the expected MAGMOM for all species.
        sort_structure (bool): Whether to sort the structure (using the default sort
            order of electronegativity) before generating input files. Defaults to True,
            the behavior you would want most of the time. This ensures that similar
            atomic species are grouped together.
        user_potcar_functional (str): Functional to use. Default (None) is to use the
            functional in the config dictionary. Valid values: "PBE", "PBE_52",
            "PBE_54", "LDA", "LDA_52", "LDA_54", "PW91", "LDA_US", "PW91_US".
        force_gamma (bool): Force gamma centered kpoint generation. Default (False) is
            to use the Automatic Density kpoint scheme, which will use the Gamma
            centered generation scheme for hexagonal cells, and Monkhorst-Pack otherwise.
        reduce_structure (None/str): Before generating the input files, generate the
            reduced structure. Default (None), does not alter the structure. Valid
            values: None, "niggli", "LLL".
        vdw: Adds default parameters for van-der-Waals functionals supported by VASP to
            INCAR. Supported functionals are: DFT-D2, undamped DFT-D3, DFT-D3 with
            Becke-Jonson damping, Tkatchenko-Scheffler, Tkatchenko-Scheffler with
            iterative Hirshfeld partitioning, MBD@rSC, dDsC, Dion's vdW-DF, DF2, optPBE,
            optB88, optB86b and rVV10.
        use_structure_charge (bool): If set to True, then the overall charge of the
            structure (structure.charge) is used to set the NELECT variable in the
            INCAR. Default is False.
        standardize (float): Whether to standardize to a primitive standard cell.
            Defaults to False.
        sym_prec (float): Tolerance for symmetry finding.
        international_monoclinic (bool): Whether to use international convention (vs
            Curtarolo) for monoclinic. Defaults True.
        validate_magmom (bool): Ensure that the missing magmom values are filled in with
            the VASP default value of 1.0.
        inherit_incar (bool): Whether to inherit INCAR settings from previous
            calculation. This might be useful to port Custodian fixes to child jobs but
            can also be dangerous e.g. when switching from GGA to meta-GGA or relax to
            static jobs. Defaults to True.
        auto_ismear (bool): If true, the values for ISMEAR and SIGMA will be set
            automatically depending on the bandgap of the system. If the bandgap is not
            known (e.g., there is no previous VASP directory) then ISMEAR=0 and
            SIGMA=0.2; if the bandgap is zero (a metallic system) then ISMEAR=2 and
            SIGMA=0.2; if the system is an insulator, then ISMEAR=-5 (tetrahedron
            smearing). Note, this only works when generating the input set from a
            previous VASP directory.
        bandgap_tol (float): Tolerance for determining if a system is metallic when
            KSPACING is set to "auto". If the bandgap is less than this value, the
            system is considered metallic. Defaults to 1e-4 (eV).
        bandgap (float): Used for determining KSPACING if KSPACING == "auto" or
            ISMEAR if auto_ismear == True. Set automatically when using from_prev_calc.
        prev_incar (str or dict): Previous INCAR used for setting parent INCAR when
            inherit_incar == True. Set automatically when using from_prev_calc.
        prev_kpoints (str or Kpoints): Previous Kpoints. Set automatically when using
            from_prev_calc.
    """
    structure: Structure | None = None
    config_dict: dict = field(default_factory=dict)
    files_to_transfer: dict = field(default_factory=dict)
    user_incar_settings: dict = field(default_factory=dict)
    user_kpoints_settings: dict = field(default_factory=dict)
    user_potcar_settings: dict = field(default_factory=dict)
    constrain_total_magmom: bool = False
    sort_structure: bool = True
    user_potcar_functional: UserPotcarFunctional = None
    force_gamma: bool = False
    reduce_structure: Literal['niggli', 'LLL'] | None = None
    vdw: str | None = None
    use_structure_charge: bool = False
    standardize: bool = False
    sym_prec: float = 0.1
    international_monoclinic: bool = True
    validate_magmom: bool = True
    inherit_incar: bool | list[str] = False
    auto_ismear: bool = False
    bandgap_tol: float = 0.0001
    bandgap: float | None = None
    prev_incar: str | dict | None = None
    prev_kpoints: str | Kpoints | None = None

    def __post_init__(self):
        """Perform validation"""
        if (valid_potcars := self._valid_potcars) and self.user_potcar_functional not in valid_potcars:
            raise ValueError(f'Invalid self.user_potcar_functional={self.user_potcar_functional!r}, must be one of {valid_potcars}')
        if hasattr(self, 'CONFIG'):
            self.config_dict = self.CONFIG
        self._config_dict = deepcopy(self.config_dict)
        self.user_incar_settings = self.user_incar_settings or {}
        self.user_kpoints_settings = self.user_kpoints_settings or {}
        self.vdw = self.vdw.lower() if isinstance(self.vdw, str) else self.vdw
        if self.user_incar_settings.get('KSPACING') and self.user_kpoints_settings is not None:
            warnings.warn('You have specified KSPACING and also supplied kpoints settings. KSPACING only has effect when there is no KPOINTS file. Since both settings were given, pymatgenwill generate a KPOINTS file and ignore KSPACING.Remove the `user_kpoints_settings` argument to enable KSPACING.', BadInputSetWarning)
        if self.vdw:
            vdw_par = loadfn(MODULE_DIR / 'vdW_parameters.yaml')
            try:
                self._config_dict['INCAR'].update(vdw_par[self.vdw])
            except KeyError:
                raise KeyError(f'Invalid or unsupported van-der-Waals functional. Supported functionals are {', '.join(vdw_par)}.')
        self.user_potcar_functional: UserPotcarFunctional = self.user_potcar_functional or self._config_dict.get('POTCAR_FUNCTIONAL', 'PBE')
        if self.user_potcar_functional != self._config_dict.get('POTCAR_FUNCTIONAL', 'PBE'):
            warnings.warn('Overriding the POTCAR functional is generally not recommended  as it significantly affect the results of calculations and compatibility with other calculations done with the same input set. Note that some POTCAR symbols specified in the configuration file may not be available in the selected functional.', BadInputSetWarning)
        if self.user_potcar_settings:
            warnings.warn('Overriding POTCARs is generally not recommended as it significantly affect the results of calculations and compatibility with other calculations done with the same input set. In many instances, it is better to write a subclass of a desired input set and override the POTCAR in the subclass to be explicit on the differences.', BadInputSetWarning)
            for key, val in self.user_potcar_settings.items():
                self._config_dict['POTCAR'][key] = val
        if not isinstance(self.structure, Structure):
            self._structure = None
        else:
            self.structure = self.structure
        if isinstance(self.prev_incar, (Path, str)):
            self.prev_incar = Incar.from_file(self.prev_incar)
        if isinstance(self.prev_kpoints, (Path, str)):
            self.prev_kpoints = Kpoints.from_file(self.prev_kpoints)
        self.prev_vasprun = None
        self.prev_outcar = None

    @property
    def structure(self) -> Structure:
        """Structure"""
        return self._structure

    @structure.setter
    def structure(self, structure: Structure | None) -> None:
        if not hasattr(self, '_config_dict'):
            self._structure = structure
            return
        if isinstance(structure, SiteCollection):
            if self.user_potcar_functional == 'PBE_54' and 'W' in structure.symbol_set:
                self.user_potcar_settings = {'W': 'W_sv', **(self.user_potcar_settings or {})}
            if self.reduce_structure:
                structure = structure.get_reduced_structure(self.reduce_structure)
            if self.sort_structure:
                structure = structure.get_sorted_structure()
            if self.validate_magmom:
                get_valid_magmom_struct(structure, spin_mode='auto')
            struct_has_Yb = any((specie.symbol == 'Yb' for site in structure for specie in site.species))
            potcar_settings = self._config_dict.get('POTCAR', {})
            if self.user_potcar_settings:
                potcar_settings.update(self.user_potcar_settings)
            uses_Yb_2_psp = potcar_settings.get('Yb', None) == 'Yb_2'
            if struct_has_Yb and uses_Yb_2_psp:
                warnings.warn('The structure contains Ytterbium (Yb) and this InputSet uses the Yb_2 PSP.\nYb_2 is known to often give bad results since Yb has oxidation state 3+ in most compounds.\nSee https://github.com/materialsproject/pymatgen/issues/2968 for details.', BadInputSetWarning)
            if self.standardize and self.sym_prec:
                structure = standardize_structure(structure, sym_prec=self.sym_prec, international_monoclinic=self.international_monoclinic)
        self._structure = structure

    def get_input_set(self, structure: Structure | None=None, prev_dir: str | Path | None=None, potcar_spec: bool=False) -> VaspInput:
        """
        Get a VASP input set.

        Note, if both ``structure`` and ``prev_dir`` are set, then the structure
        specified will be preferred over the final structure from the last VASP run.

        Args:
            structure (Structure): A structure.
            prev_dir (str or Path): A previous directory to generate the input set from.
            potcar_spec (bool): Instead of generating a Potcar object, use a list of
                potcar symbols. This will be written as a "POTCAR.spec" file. This is
                intended to help sharing an input set with people who might not have a
                license to specific Potcar files. Given a "POTCAR.spec", the specific
                POTCAR file can be re-generated using pymatgen with the
                "generate_potcar" function in the pymatgen CLI.

        Returns:
            VaspInput: A VASP input object.
        """
        if structure is None and prev_dir is None and (self.structure is None):
            raise ValueError('Either structure or prev_dir must be set.')
        self._set_previous(prev_dir)
        if structure is not None:
            self.structure = structure
        return VaspInput(incar=self.incar, kpoints=self.kpoints, poscar=self.poscar, potcar=self.potcar_symbols if potcar_spec else self.potcar)

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR config for this calculation type."""
        return {}

    @property
    def kpoints_updates(self) -> dict | Kpoints:
        """Get updates to the kpoints configuration for this calculation type.

        Note, these updates will be ignored if the user has set user_kpoint_settings.

        Returns:
            dict or Kpoints: A dictionary of updates to apply to the KPOINTS config
                or a Kpoints object.
        """
        return {}

    def _set_previous(self, prev_dir: str | Path | None=None):
        """Load previous calculation outputs."""
        if prev_dir is not None:
            vasprun, outcar = get_vasprun_outcar(prev_dir)
            self.prev_vasprun = vasprun
            self.prev_outcar = outcar
            self.prev_incar = vasprun.incar
            self.prev_kpoints = Kpoints.from_dict(vasprun.kpoints.as_dict())
            if vasprun.efermi is None:
                vasprun.efermi = outcar.efermi
            bs = vasprun.get_band_structure(efermi='smart')
            self.bandgap = 0 if bs.is_metal() else bs.get_band_gap()['energy']
            self.structure = get_structure_from_prev_run(vasprun, outcar)

    @property
    def incar(self) -> Incar:
        """Get the INCAR."""
        if self.structure is None:
            raise RuntimeError('No structure is associated with the input set!')
        prev_incar: dict[str, Any] = {}
        if self.inherit_incar is True and self.prev_incar:
            prev_incar = self.prev_incar
        elif isinstance(self.inherit_incar, (list, tuple)) and self.prev_incar:
            prev_incar = {k: self.prev_incar[k] for k in self.inherit_incar if k in self.prev_incar}
        incar_updates = self.incar_updates
        settings = dict(self._config_dict['INCAR'])
        auto_updates = {}
        _apply_incar_updates(settings, incar_updates)
        _apply_incar_updates(settings, self.user_incar_settings)
        structure = self.structure
        comp = structure.composition
        elements = sorted((el for el in comp.elements if comp[el] > 0), key=lambda e: e.X)
        most_electro_neg = elements[-1].symbol
        poscar = Poscar(structure)
        hubbard_u = settings.get('LDAU', False)
        incar = Incar()
        for k, v in settings.items():
            if k == 'MAGMOM':
                mag = []
                for site in structure:
                    if hasattr(site, 'magmom'):
                        mag.append(site.magmom)
                    elif getattr(site.specie, 'spin', None) is not None:
                        mag.append(site.specie.spin)
                    elif str(site.specie) in v:
                        if site.specie.symbol == 'Co' and v[str(site.specie)] <= 1.0:
                            warnings.warn('Co without an oxidation state is initialized as low spin by default in Pymatgen. If this default behavior is not desired, please set the spin on the magmom on the site directly to ensure correct initialization.')
                        mag.append(v.get(str(site.specie)))
                    else:
                        if site.specie.symbol == 'Co':
                            warnings.warn('Co without an oxidation state is initialized as low spin by default in Pymatgen. If this default behavior is not desired, please set the spin on the magmom on the site directly to ensure correct initialization.')
                        mag.append(v.get(site.specie.symbol, 0.6))
                incar[k] = mag
            elif k in ('LDAUU', 'LDAUJ', 'LDAUL'):
                if hubbard_u:
                    if hasattr(structure[0], k.lower()):
                        m = {site.specie.symbol: getattr(site, k.lower()) for site in structure}
                        incar[k] = [m[sym] for sym in poscar.site_symbols]
                    elif most_electro_neg in v and isinstance(v[most_electro_neg], dict):
                        incar[k] = [v[most_electro_neg].get(sym, 0) for sym in poscar.site_symbols]
                    else:
                        incar[k] = [v.get(sym, 0) if isinstance(v.get(sym, 0), (float, int)) else 0 for sym in poscar.site_symbols]
            elif k.startswith('EDIFF') and k != 'EDIFFG':
                if 'EDIFF' not in settings and k == 'EDIFF_PER_ATOM':
                    incar['EDIFF'] = float(v) * len(structure)
                else:
                    incar['EDIFF'] = float(settings['EDIFF'])
            elif k == 'KSPACING' and v == 'auto':
                bandgap = 0 if self.bandgap is None else self.bandgap
                incar[k] = auto_kspacing(bandgap, self.bandgap_tol)
            else:
                incar[k] = v
        has_u = hubbard_u and sum(incar['LDAUU']) > 0
        if not has_u:
            for key in list(incar):
                if key.startswith('LDAU'):
                    del incar[key]
        if 'LMAXMIX' not in settings:
            if any((el.Z > 56 for el in structure.composition)):
                incar['LMAXMIX'] = 6
            elif any((el.Z > 20 for el in structure.composition)):
                incar['LMAXMIX'] = 4
        if not incar.get('LASPH', False) and (incar.get('METAGGA') or incar.get('LHFCALC', False) or incar.get('LDAU', False) or incar.get('LUSE_VDW', False)):
            warnings.warn('LASPH = True should be set for +U, meta-GGAs, hybrids, and vdW-DFT', BadInputSetWarning)
        skip = list(self.user_incar_settings) + list(incar_updates)
        skip += ['MAGMOM', 'NUPDOWN', 'LDAUU', 'LDAUL', 'LDAUJ']
        _apply_incar_updates(incar, prev_incar, skip=skip)
        if self.constrain_total_magmom:
            nupdown = sum((mag if abs(mag) > 0.6 else 0 for mag in incar['MAGMOM']))
            if abs(nupdown - round(nupdown)) > 1e-05:
                warnings.warn('constrain_total_magmom was set to True, but the sum of MAGMOM values is not an integer. NUPDOWN is meant to set the spin multiplet and should typically be an integer. You are likely better off changing the values of MAGMOM or simply setting NUPDOWN directly in your INCAR settings.', UserWarning, stacklevel=1)
            auto_updates['NUPDOWN'] = nupdown
        if self.use_structure_charge:
            auto_updates['NELECT'] = self.nelect
        if incar.get('LHFCALC', False) is True and incar.get('ALGO', 'Normal') not in ['Normal', 'All', 'Damped']:
            warnings.warn('Hybrid functionals only support Algo = All, Damped, or Normal.', BadInputSetWarning)
        if self.auto_ismear:
            if self.bandgap is None:
                auto_updates.update(ISMEAR=2, SIGMA=0.2)
            elif self.bandgap <= self.bandgap_tol:
                auto_updates.update(ISMEAR=2, SIGMA=0.2)
            else:
                auto_updates.update(ISMEAR=-5, SIGMA=0.05)
        kpoints = self.kpoints
        if kpoints is not None:
            incar.pop('KSPACING', None)
        elif 'KSPACING' in incar and 'KSPACING' not in self.user_incar_settings and ('KSPACING' in prev_incar):
            incar['KSPACING'] = prev_incar['KSPACING']
        _apply_incar_updates(incar, auto_updates, skip=list(self.user_incar_settings))
        _remove_unused_incar_params(incar, skip=list(self.user_incar_settings))
        if kpoints is not None and np.prod(kpoints.kpts) < 4 and (incar.get('ISMEAR', 0) == -5):
            incar['ISMEAR'] = 0
        if self.user_incar_settings.get('KSPACING', 0) > 0.5 and incar.get('ISMEAR', 0) == -5:
            warnings.warn('Large KSPACING value detected with ISMEAR = -5. Ensure that VASP generates an adequate number of KPOINTS, lower KSPACING, or set ISMEAR = 0', BadInputSetWarning)
        ismear = incar.get('ISMEAR', 1)
        sigma = incar.get('SIGMA', 0.2)
        if all((elem.is_metal for elem in structure.composition)) and incar.get('NSW', 0) > 0 and (ismear < 0 or (ismear == 0 and sigma > 0.05)):
            ismear_docs = 'https://www.vasp.at/wiki/index.php/ISMEAR'
            msg = ''
            if ismear < 0:
                msg = f'Relaxation of likely metal with ISMEAR < 0 ({ismear}).'
            elif ismear == 0 and sigma > 0.05:
                msg = f'ISMEAR = 0 with a small SIGMA ({sigma}) detected.'
            warnings.warn(f'{msg} See VASP recommendations on ISMEAR for metals ({ismear_docs}).', BadInputSetWarning, stacklevel=1)
        return incar

    @property
    def poscar(self) -> Poscar:
        """Poscar"""
        if self.structure is None:
            raise RuntimeError('No structure is associated with the input set!')
        return Poscar(self.structure)

    @property
    def potcar_functional(self) -> UserPotcarFunctional:
        """Returns the functional used for POTCAR generation."""
        return self.user_potcar_functional

    @property
    def nelect(self) -> float:
        """Gets the default number of electrons for a given structure."""
        if self.structure is None:
            raise RuntimeError('No structure is associated with the input set!')
        n_electrons_by_element = {p.element: p.nelectrons for p in self.potcar}
        n_elect = sum((num_atoms * n_electrons_by_element[el.symbol] for el, num_atoms in self.structure.composition.items()))
        if self.use_structure_charge:
            return n_elect - self.structure.charge
        return n_elect

    @property
    def kpoints(self) -> Kpoints | None:
        """Get the kpoints file."""
        if self.structure is None:
            raise RuntimeError('No structure is associated with the input set!')
        if (self.user_incar_settings.get('KSPACING', None) is not None or self.incar_updates.get('KSPACING', None) is not None or self._config_dict['INCAR'].get('KSPACING', None) is not None) and self.user_kpoints_settings == {}:
            return None
        kpoints_updates = self.kpoints_updates
        if self.user_kpoints_settings != {}:
            kconfig = deepcopy(self.user_kpoints_settings)
        elif isinstance(kpoints_updates, Kpoints):
            return kpoints_updates
        elif kpoints_updates != {}:
            kconfig = kpoints_updates
        else:
            kconfig = deepcopy(self._config_dict.get('KPOINTS', {}))
        if isinstance(kconfig, Kpoints):
            return kconfig
        explicit = kconfig.get('explicit') or len(kconfig.get('added_kpoints', [])) > 0 or 'zero_weighted_reciprocal_density' in kconfig or ('zero_weighted_line_density' in kconfig)
        if kconfig.get('length'):
            if explicit:
                raise ValueError('length option cannot be used with explicit k-point generation, added_kpoints, or zero weighted k-points.')
            return Kpoints.automatic(kconfig['length'])
        base_kpoints = None
        if kconfig.get('line_density'):
            kpath = HighSymmKpath(self.structure, **kconfig.get('kpath_kwargs', {}))
            frac_k_points, k_points_labels = kpath.get_kpoints(line_density=kconfig['line_density'], coords_are_cartesian=False)
            base_kpoints = Kpoints(comment='Non SCF run along symmetry lines', style=Kpoints.supported_modes.Reciprocal, num_kpts=len(frac_k_points), kpts=frac_k_points, labels=k_points_labels, kpts_weights=[1] * len(frac_k_points))
        elif kconfig.get('grid_density') or kconfig.get('reciprocal_density'):
            if kconfig.get('grid_density'):
                base_kpoints = Kpoints.automatic_density(self.structure, int(kconfig['grid_density']), self.force_gamma)
            elif kconfig.get('reciprocal_density'):
                density = kconfig['reciprocal_density']
                base_kpoints = Kpoints.automatic_density_by_vol(self.structure, density, self.force_gamma)
            if explicit:
                sga = SpacegroupAnalyzer(self.structure, symprec=self.sym_prec)
                mesh = sga.get_ir_reciprocal_mesh(base_kpoints.kpts[0])
                base_kpoints = Kpoints(comment='Uniform grid', style=Kpoints.supported_modes.Reciprocal, num_kpts=len(mesh), kpts=[i[0] for i in mesh], kpts_weights=[i[1] for i in mesh])
            else:
                return base_kpoints
        zero_weighted_kpoints = None
        if kconfig.get('zero_weighted_line_density'):
            kpath = HighSymmKpath(self.structure)
            frac_k_points, k_points_labels = kpath.get_kpoints(line_density=kconfig['zero_weighted_line_density'], coords_are_cartesian=False)
            zero_weighted_kpoints = Kpoints(comment='Hybrid run along symmetry lines', style=Kpoints.supported_modes.Reciprocal, num_kpts=len(frac_k_points), kpts=frac_k_points, labels=k_points_labels, kpts_weights=[0] * len(frac_k_points))
        elif kconfig.get('zero_weighted_reciprocal_density'):
            zero_weighted_kpoints = Kpoints.automatic_density_by_vol(self.structure, kconfig['zero_weighted_reciprocal_density'], self.force_gamma)
            sga = SpacegroupAnalyzer(self.structure, symprec=self.sym_prec)
            mesh = sga.get_ir_reciprocal_mesh(zero_weighted_kpoints.kpts[0])
            zero_weighted_kpoints = Kpoints(comment='Uniform grid', style=Kpoints.supported_modes.Reciprocal, num_kpts=len(mesh), kpts=[i[0] for i in mesh], kpts_weights=[0 for i in mesh])
        added_kpoints = None
        if kconfig.get('added_kpoints'):
            points: list = kconfig.get('added_kpoints')
            added_kpoints = Kpoints(comment='Specified k-points only', style=Kpoints.supported_modes.Reciprocal, num_kpts=len(points), kpts=points, labels=['user-defined'] * len(points), kpts_weights=[0] * len(points))
        if base_kpoints and (not (added_kpoints or zero_weighted_kpoints)):
            return base_kpoints
        if added_kpoints and (not (base_kpoints or zero_weighted_kpoints)):
            return added_kpoints
        if 'line_density' in kconfig and zero_weighted_kpoints:
            raise ValueError('Cannot combine line_density and zero weighted k-points options')
        if zero_weighted_kpoints and (not base_kpoints):
            raise ValueError('Zero weighted k-points must be used with reciprocal_density or grid_density options')
        if not (base_kpoints or zero_weighted_kpoints or added_kpoints):
            raise ValueError("Invalid k-point generation algo. Supported Keys are 'grid_density' for Kpoints.automatic_density generation, 'reciprocal_density' for KPoints.automatic_density_by_vol generation, 'length' for Kpoints.automatic generation, 'line_density' for line mode generation, 'added_kpoints' for specific k-points to include,  'zero_weighted_reciprocal_density' for a zero weighted uniform mesh, or 'zero_weighted_line_density' for a zero weighted line mode mesh.")
        return _combine_kpoints(base_kpoints, zero_weighted_kpoints, added_kpoints)

    @property
    def potcar(self) -> Potcar:
        """Potcar object"""
        if self.structure is None:
            raise RuntimeError('No structure is associated with the input set!')
        return super().potcar

    def estimate_nbands(self) -> int:
        """
        Estimate the number of bands that VASP will initialize a
        calculation with by default. Note that in practice this
        can depend on # of cores (if not set explicitly).
        Note that this formula is slightly different than the formula on the VASP wiki
        (as of July 2023). This is because the formula in the source code (`main.F`) is
        slightly different than what is on the wiki.
        """
        if self.structure is None:
            raise RuntimeError('No structure is associated with the input set!')
        n_ions = len(self.structure)
        if self.incar['ISPIN'] == 1:
            n_mag = 0
        else:
            n_mag = sum(self.incar['MAGMOM'])
            n_mag = np.floor((n_mag + 1) / 2)
        possible_val_1 = np.floor((self.nelect + 2) / 2) + max(np.floor(n_ions / 2), 3)
        possible_val_2 = np.floor(self.nelect * 0.6)
        n_bands = max(possible_val_1, possible_val_2) + n_mag
        if self.incar.get('LNONCOLLINEAR') is True:
            n_bands = n_bands * 2
        if (n_par := self.incar.get('NPAR')):
            n_bands = np.floor((n_bands + n_par - 1) / n_par) * n_par
        return int(n_bands)

    def override_from_prev_calc(self, prev_calc_dir='.'):
        """
        Update the input set to include settings from a previous calculation.

        Args:
            prev_calc_dir (str): The path to the previous calculation directory.

        Returns:
            The input set with the settings (structure, k-points, incar, etc)
            updated using the previous VASP run.
        """
        self._set_previous(prev_calc_dir)
        if self.standardize:
            warnings.warn('Use of standardize=True with from_prev_run is not recommended as there is no guarantee the copied files will be appropriate for the standardized structure.')
        files_to_transfer = {}
        if getattr(self, 'copy_chgcar', False):
            chgcars = sorted(glob(str(Path(prev_calc_dir) / 'CHGCAR*')))
            if chgcars:
                files_to_transfer['CHGCAR'] = str(chgcars[-1])
        if getattr(self, 'copy_wavecar', False):
            for fname in ('WAVECAR', 'WAVEDER', 'WFULL'):
                wavecar_files = sorted(glob(str(Path(prev_calc_dir) / (fname + '*'))))
                if wavecar_files:
                    if fname == 'WFULL':
                        for wavecar_file in wavecar_files:
                            fname = Path(wavecar_file).name
                            fname = fname.split('.')[0]
                            files_to_transfer[fname] = wavecar_file
                    else:
                        files_to_transfer[fname] = str(wavecar_files[-1])
        self.files_to_transfer.update(files_to_transfer)
        return self

    @classmethod
    def from_prev_calc(cls, prev_calc_dir: str, **kwargs) -> Self:
        """
        Generate a set of VASP input files for static calculations from a
        directory of previous VASP run.

        Args:
            prev_calc_dir (str): Directory containing the outputs(
                vasprun.xml and OUTCAR) of previous vasp run.
            **kwargs: All kwargs supported by MPStaticSet, other than prev_incar
                and prev_structure and prev_kpoints which are determined from
                the prev_calc_dir.
        """
        input_set = cls(_dummy_structure, **kwargs)
        return input_set.override_from_prev_calc(prev_calc_dir=prev_calc_dir)

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return type(self).__name__

    def write_input(self, output_dir: str, make_dir_if_not_present: bool=True, include_cif: bool=False, potcar_spec: bool=False, zip_output: bool=False):
        """
        Writes out all input to a directory.

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
            zip_output (bool): Whether to zip each VASP input file written to the output directory.
        """
        super().write_input(output_dir=output_dir, make_dir_if_not_present=make_dir_if_not_present, include_cif=include_cif, potcar_spec=potcar_spec, zip_output=zip_output)
        for k, v in self.files_to_transfer.items():
            with zopen(v, 'rb') as fin, zopen(str(Path(output_dir) / k), 'wb') as fout:
                shutil.copyfileobj(fin, fout)

    def calculate_ng(self, max_prime_factor: int=7, must_inc_2: bool=True, custom_encut: float | None=None, custom_prec: str | None=None) -> tuple:
        """
        Calculates the NGX, NGY, and NGZ values using the information available in the INCAR and POTCAR
        This is meant to help with making initial guess for the FFT grid so we can interact with the Charge density API.

        Args:
            max_prime_factor (int): the valid prime factors of the grid size in each direction
                VASP has many different setting for this to handle many compiling options.
                For typical MPI options all prime factors up to 7 are allowed
            must_inc_2 (bool): Whether 2 must be a prime factor of the result. Defaults to True.
            custom_encut (float | None): Calculates the FFT grid parameters using a custom
                ENCUT that may be different from what is generated by the input set. Defaults to None.
                Do *not* use this unless you know what you are doing.
            custom_prec (str | None): Calculates the FFT grid parameters using a custom prec
                that may be different from what is generated by the input set. Defaults to None.
                Do *not* use this unless you know what you are doing.
        """
        _RYTOEV = 13.605826
        _AUTOA = 0.529177249
        if custom_encut is not None:
            encut = custom_encut
        elif self.incar.get('ENCUT', 0) > 0:
            encut = self.incar['ENCUT']
        else:
            encut = max((i_species.enmax for i_species in self.get_vasp_input()['POTCAR']))
        PREC = self.incar.get('PREC', 'Normal') if custom_prec is None else custom_prec
        if PREC[0].lower() in {'l', 'm', 'h'}:
            raise NotImplementedError('PREC = LOW/MEDIUM/HIGH from VASP 4.x and not supported, Please use NORMA/SINGLE/ACCURATE')
        if PREC[0].lower() not in {'a', 's', 'n', 'l', 'm', 'h'}:
            raise ValueError(f'PREC={PREC!r} does not exist. If this is no longer correct, please update this code.')
        CUTOFF = [np.sqrt(encut / _RYTOEV) / (2 * np.pi / (anorm / _AUTOA)) for anorm in self.poscar.structure.lattice.abc]
        _WFACT = 4 if PREC[0].lower() in {'a', 's'} else 3

        def next_g_size(cur_g_size):
            g_size = int(_WFACT * cur_g_size + 0.5)
            return next_num_with_prime_factors(g_size, max_prime_factor, must_inc_2)
        ng_vec = [*map(next_g_size, CUTOFF)]
        finer_g_scale = 2 if PREC[0].lower() in {'a', 'n'} else 1
        return (ng_vec, [ng_ * finer_g_scale for ng_ in ng_vec])
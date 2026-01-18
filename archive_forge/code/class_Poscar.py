from __future__ import annotations
import codecs
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import re
import subprocess
import warnings
from collections import namedtuple
from enum import Enum, unique
from glob import glob
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
import scipy.constants as const
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from monty.os import cd
from monty.os.path import zpath
from monty.serialization import dumpfn, loadfn
from tabulate import tabulate
from pymatgen.core import SETTINGS, Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
class Poscar(MSONable):
    """Object for representing the data in a POSCAR or CONTCAR file.

    Attributes:
        structure: Associated Structure.
        comment: Optional comment string.
        true_names: Boolean indication whether Poscar contains actual real names parsed
            from either a POTCAR or the POSCAR itself.
        selective_dynamics: Selective dynamics attribute for each site if available.
            A Nx3 array of booleans.
        velocities: Velocities for each site (typically read in from a CONTCAR).
            A Nx3 array of floats.
        predictor_corrector: Predictor corrector coordinates and derivatives for each site;
            i.e. a list of three 1x3 arrays for each site (typically read in from an MD CONTCAR).
        predictor_corrector_preamble: Predictor corrector preamble contains the predictor-corrector key,
            POTIM, and thermostat parameters that precede the site-specific predictor corrector data in MD CONTCAR.
        lattice_velocities: Lattice velocities and current lattice (typically read
            in from an MD CONTCAR). A 6x3 array of floats.
        temperature: Temperature of velocity Maxwell-Boltzmann initialization.
            Initialized to -1 (MB hasn't been performed).
    """

    def __init__(self, structure: Structure, comment: str | None=None, selective_dynamics: ArrayLike | None=None, true_names: bool=True, velocities: ArrayLike | None=None, predictor_corrector: ArrayLike | None=None, predictor_corrector_preamble: str | None=None, lattice_velocities: ArrayLike | None=None, sort_structure: bool=False):
        """
        Args:
            structure (Structure): Structure object.
            comment (str | None, optional): Optional comment line for POSCAR. Defaults to unit
                cell formula of structure. Defaults to None.
            selective_dynamics (ArrayLike | None, optional): Bool values for selective dynamics,
                where N is the number of sites. Defaults to None.
            true_names (bool, optional): Set to False if the names in the POSCAR are not
                well-defined and ambiguous. This situation arises commonly in
                VASP < 5 where the POSCAR sometimes does not contain element
                symbols. Defaults to True.
            velocities (ArrayLike | None, optional): Velocities for the POSCAR. Typically parsed
                in MD runs or can be used to initialize velocities. Defaults to None.
            predictor_corrector (ArrayLike | None, optional): Predictor corrector for the POSCAR.
                Typically parsed in MD runs. Defaults to None.
            predictor_corrector_preamble (str | None, optional): Preamble to the predictor
                corrector. Defaults to None.
            lattice_velocities (ArrayLike | None, optional): Lattice velocities and current
                lattice for the POSCAR. Available in MD runs with variable cell. Defaults to None.
            sort_structure (bool, optional): Whether to sort the structure. Useful if species
                are not grouped properly together. Defaults to False.
        """
        if structure.is_ordered:
            site_properties = {}
            if selective_dynamics is not None:
                selective_dynamics = np.array(selective_dynamics)
                if not selective_dynamics.all():
                    site_properties['selective_dynamics'] = selective_dynamics
            if velocities:
                velocities = np.array(velocities)
                if velocities.any():
                    site_properties['velocities'] = velocities
            if predictor_corrector:
                predictor_corrector = np.array(predictor_corrector)
                if predictor_corrector.any():
                    site_properties['predictor_corrector'] = predictor_corrector
            structure = Structure.from_sites(structure)
            self.structure = structure.copy(site_properties=site_properties)
            if sort_structure:
                self.structure = self.structure.get_sorted_structure()
            self.true_names = true_names
            self.comment = structure.formula if comment is None else comment
            if predictor_corrector_preamble:
                self.structure.properties['predictor_corrector_preamble'] = predictor_corrector_preamble
            if lattice_velocities and np.any(lattice_velocities):
                self.structure.properties['lattice_velocities'] = np.asarray(lattice_velocities)
        else:
            raise ValueError('Disordered structure with partial occupancies cannot be converted into POSCAR!')
        self.temperature = -1.0

    @property
    def velocities(self):
        """Velocities in Poscar."""
        return self.structure.site_properties.get('velocities')

    @property
    def selective_dynamics(self):
        """Selective dynamics in Poscar."""
        return self.structure.site_properties.get('selective_dynamics')

    @property
    def predictor_corrector(self):
        """Predictor corrector in Poscar."""
        return self.structure.site_properties.get('predictor_corrector')

    @property
    def predictor_corrector_preamble(self):
        """Predictor corrector preamble in Poscar."""
        return self.structure.properties.get('predictor_corrector_preamble')

    @property
    def lattice_velocities(self):
        """Lattice velocities in Poscar (including the current lattice vectors)."""
        return self.structure.properties.get('lattice_velocities')

    @velocities.setter
    def velocities(self, velocities):
        """Setter for Poscar.velocities."""
        self.structure.add_site_property('velocities', velocities)

    @selective_dynamics.setter
    def selective_dynamics(self, selective_dynamics):
        """Setter for Poscar.selective_dynamics."""
        self.structure.add_site_property('selective_dynamics', selective_dynamics)

    @predictor_corrector.setter
    def predictor_corrector(self, predictor_corrector):
        """Setter for Poscar.predictor_corrector."""
        self.structure.add_site_property('predictor_corrector', predictor_corrector)

    @predictor_corrector_preamble.setter
    def predictor_corrector_preamble(self, predictor_corrector_preamble):
        """Setter for Poscar.predictor_corrector."""
        self.structure.properties['predictor_corrector'] = predictor_corrector_preamble

    @lattice_velocities.setter
    def lattice_velocities(self, lattice_velocities: ArrayLike) -> None:
        """Setter for Poscar.lattice_velocities."""
        self.structure.properties['lattice_velocities'] = np.asarray(lattice_velocities)

    @property
    def site_symbols(self) -> list[str]:
        """
        Sequence of symbols associated with the Poscar. Similar to 6th line in VASP 5+ POSCAR.
        """
        syms = [site.specie.symbol for site in self.structure]
        return [a[0] for a in itertools.groupby(syms)]

    @property
    def natoms(self) -> list[int]:
        """
        Sequence of number of sites of each type associated with the Poscar.
        Similar to 7th line in vasp 5+ POSCAR or the 6th line in vasp 4 POSCAR.
        """
        syms = [site.specie.symbol for site in self.structure]
        return [len(tuple(a[1])) for a in itertools.groupby(syms)]

    def __setattr__(self, name, value):
        if name in ('selective_dynamics', 'velocities') and value is not None and (len(value) > 0):
            value = np.array(value)
            dim = value.shape
            if dim[1] != 3 or dim[0] != len(self.structure):
                raise ValueError(f'{name} array must be same length as the structure.')
            value = value.tolist()
        super().__setattr__(name, value)

    @classmethod
    def from_file(cls, filename, check_for_potcar=True, read_velocities=True, **kwargs) -> Self:
        """
        Reads a Poscar from a file.

        The code will try its best to determine the elements in the POSCAR in
        the following order:

        1. If check_for_potcar is True, the code will try to check if a POTCAR
        is in the same directory as the POSCAR and use elements from that by
        default. (This is the VASP default sequence of priority).
        2. If the input file is VASP5-like and contains element symbols in the
        6th line, the code will use that if check_for_potcar is False or there
        is no POTCAR found.
        3. Failing (2), the code will check if a symbol is provided at the end
        of each coordinate.

        If all else fails, the code will just assign the first n elements in
        increasing atomic number, where n is the number of species, to the
        Poscar, where a warning would be issued. For example, H, He, Li, ....
        This will ensure at least a unique element is assigned to each site and
        any analysis that does not require specific elemental properties should work.

        Args:
            filename (str): File name containing Poscar data.
            check_for_potcar (bool): Whether to check if a POTCAR is present
                in the same directory as the POSCAR. Defaults to True.
            read_velocities (bool): Whether to read or not velocities if they
                are present in the POSCAR. Default is True.

        Returns:
            Poscar object.
        """
        if 'check_for_POTCAR' in kwargs:
            warnings.warn('check_for_POTCAR is deprecated. Use check_for_potcar instead.', DeprecationWarning)
            check_for_potcar = kwargs.pop('check_for_POTCAR')
        dirname = os.path.dirname(os.path.abspath(filename))
        names = None
        if check_for_potcar and SETTINGS.get('PMG_POTCAR_CHECKS') is not False:
            potcars = glob(f'{dirname}/*POTCAR*')
            if potcars:
                try:
                    potcar = Potcar.from_file(sorted(potcars)[0])
                    names = [sym.split('_')[0] for sym in potcar.symbols]
                    [get_el_sp(n) for n in names]
                except Exception:
                    names = None
        with zopen(filename, mode='rt') as file:
            return cls.from_str(file.read(), names, read_velocities=read_velocities)

    @classmethod
    def from_str(cls, data, default_names=None, read_velocities=True) -> Self:
        """
        Reads a Poscar from a string.

        The code will try its best to determine the elements in the POSCAR in
        the following order:

        1. If default_names are supplied and valid, it will use those. Usually,
        default names comes from an external source, such as a POTCAR in the
        same directory.

        2. If there are no valid default names but the input file is VASP5-like
        and contains element symbols in the 6th line, the code will use that.

        3. Failing (2), the code will check if a symbol is provided at the end
        of each coordinate.

        If all else fails, the code will just assign the first n elements in
        increasing atomic number, where n is the number of species, to the
        Poscar. For example, H, He, Li, .... This will ensure at least a
        unique element is assigned to each site and any analysis that does not
        require specific elemental properties should work fine.

        Args:
            data (str): String containing Poscar data.
            default_names ([str]): Default symbols for the POSCAR file,
                usually coming from a POTCAR in the same directory.
            read_velocities (bool): Whether to read or not velocities if they
                are present in the POSCAR. Default is True.

        Returns:
            Poscar object.
        """
        chunks: list[str] = re.split('\\n\\s*\\n', data.rstrip(), flags=re.MULTILINE)
        try:
            if chunks[0] == '':
                chunks.pop(0)
                chunks[0] = '\n' + chunks[0]
        except IndexError:
            raise ValueError('Empty POSCAR')
        lines = list(clean_lines(chunks[0].split('\n'), remove_empty_lines=False))
        comment = lines[0]
        scale = float(lines[1])
        lattice = np.array([[float(i) for i in line.split()] for line in lines[2:5]])
        if scale < 0:
            vol = abs(np.linalg.det(lattice))
            lattice *= (-scale / vol) ** (1 / 3)
        else:
            lattice *= scale
        vasp5_symbols = False
        try:
            n_atoms = [int(i) for i in lines[5].split()]
            ipos = 6
        except ValueError:
            vasp5_symbols = True
            symbols = [symbol.split('/')[0] for symbol in lines[5].split()]
            n_lines_symbols = 1
            for n_lines_symbols in range(1, 11):
                with contextlib.suppress(ValueError):
                    int(lines[5 + n_lines_symbols].split()[0])
                    break
            for i_line_symbols in range(6, 5 + n_lines_symbols):
                symbols.extend(lines[i_line_symbols].split())
            n_atoms = []
            iline_natoms_start = 5 + n_lines_symbols
            for iline_natoms in range(iline_natoms_start, iline_natoms_start + n_lines_symbols):
                n_atoms.extend([int(i) for i in lines[iline_natoms].split()])
            atomic_symbols = []
            for i, nat in enumerate(n_atoms):
                atomic_symbols.extend([symbols[i]] * nat)
            ipos = 5 + 2 * n_lines_symbols
        pos_type = lines[ipos].split()[0]
        has_selective_dynamics = False
        if pos_type[0] in 'sS':
            has_selective_dynamics = True
            ipos += 1
            pos_type = lines[ipos].split()[0]
        cart = pos_type[0] in 'cCkK'
        n_sites = sum(n_atoms)
        if default_names:
            with contextlib.suppress(IndexError):
                atomic_symbols = []
                for i, nat in enumerate(n_atoms):
                    atomic_symbols.extend([default_names[i]] * nat)
                vasp5_symbols = True
        if not vasp5_symbols:
            ind = 6 if has_selective_dynamics else 3
            try:
                atomic_symbols = [line.split()[ind] for line in lines[ipos + 1:ipos + 1 + n_sites]]
                if not all((Element.is_valid_symbol(sym) for sym in atomic_symbols)):
                    raise ValueError('Non-valid symbols detected.')
                vasp5_symbols = True
            except (ValueError, IndexError):
                atomic_symbols = []
                for i, nat in enumerate(n_atoms, start=1):
                    sym = Element.from_Z(i).symbol
                    atomic_symbols.extend([sym] * nat)
                warnings.warn(f'Elements in POSCAR cannot be determined. Defaulting to false names {atomic_symbols}.', BadPoscarWarning)
        coords = []
        selective_dynamics: list[np.ndarray] | None = [] if has_selective_dynamics else None
        for i in range(n_sites):
            tokens = lines[ipos + 1 + i].split()
            crd_scale = scale if cart else 1
            coords.append([float(j) * crd_scale for j in tokens[:3]])
            if selective_dynamics is not None:
                if any((value not in {'T', 'F'} for value in tokens[3:6])):
                    warnings.warn("Selective dynamics values must be either 'T' or 'F'.", BadPoscarWarning)
                if atomic_symbols[i] == 'F' and len(tokens[3:]) >= 4 and ('F' in tokens[3:7]):
                    warnings.warn('Selective dynamics toggled with Fluorine element detected. Make sure the 4th-6th entry each position line is selective dynamics info.', BadPoscarWarning)
                selective_dynamics.append([value == 'T' for value in tokens[3:6]])
        if selective_dynamics is not None and all((all((i is True for i in in_list)) for in_list in selective_dynamics)):
            warnings.warn('Ignoring selective dynamics tag, as no ionic degrees of freedom were fixed.', BadPoscarWarning)
        struct = Structure(lattice, atomic_symbols, coords, to_unit_cell=False, validate_proximity=False, coords_are_cartesian=cart)
        lattice_velocities = []
        velocities = []
        predictor_corrector = []
        predictor_corrector_preamble = ''
        if read_velocities:
            if len(lines) > ipos + n_sites + 1 and lines[ipos + n_sites + 1].lower().startswith('l'):
                for line in lines[ipos + n_sites + 3:ipos + n_sites + 9]:
                    lattice_velocities.append([float(tok) for tok in line.split()])
            if len(chunks) > 1:
                for line in chunks[1].strip().split('\n'):
                    velocities.append([float(tok) for tok in line.split()])
            if len(chunks) > 2:
                lines = chunks[2].strip().split('\n')
                predictor_corrector_preamble = f'{lines[0]}\n{lines[1]}\n{lines[2]}'
                lines = lines[3:]
                for st in range(n_sites):
                    d1 = [float(tok) for tok in lines[st].split()]
                    d2 = [float(tok) for tok in lines[st + n_sites].split()]
                    d3 = [float(tok) for tok in lines[st + 2 * n_sites].split()]
                    predictor_corrector.append([d1, d2, d3])
        return cls(struct, comment, selective_dynamics, vasp5_symbols, velocities=velocities, predictor_corrector=predictor_corrector, predictor_corrector_preamble=predictor_corrector_preamble, lattice_velocities=lattice_velocities)

    def get_str(self, direct: bool=True, vasp4_compatible: bool=False, significant_figures: int=16) -> str:
        """
        Returns a string to be written as a POSCAR file. By default, site
        symbols are written, which means compatibility is for vasp >= 5.

        Args:
            direct (bool): Whether coordinates are output in direct or
                Cartesian. Defaults to True.
            vasp4_compatible (bool): Set to True to omit site symbols on 6th
                line to maintain backward vasp 4.x compatibility. Defaults
                to False.
            significant_figures (int): No. of significant figures to
                output all quantities. Defaults to 16. Note that positions are
                output in fixed point, while velocities are output in
                scientific format.

        Returns:
            String representation of POSCAR.
        """
        lattice = self.structure.lattice
        if np.linalg.det(lattice.matrix) < 0:
            lattice = Lattice(-lattice.matrix)
        format_str = f'{{:{significant_figures + 5}.{significant_figures}f}}'
        lines = [self.comment, '1.0']
        for vec in lattice.matrix:
            lines.append(' '.join((format_str.format(c) for c in vec)))
        if self.true_names and (not vasp4_compatible):
            lines.append(' '.join(self.site_symbols))
        lines.append(' '.join(map(str, self.natoms)))
        if self.selective_dynamics:
            lines.append('Selective dynamics')
        lines.append('direct' if direct else 'cartesian')
        for idx, site in enumerate(self.structure):
            coords = site.frac_coords if direct else site.coords
            line = ' '.join((format_str.format(c) for c in coords))
            if self.selective_dynamics is not None:
                sd = ['T' if j else 'F' for j in self.selective_dynamics[idx]]
                line += f' {sd[0]} {sd[1]} {sd[2]}'
            line += f' {site.species_string}'
            lines.append(line)
        if self.lattice_velocities is not None:
            try:
                lines.extend(['Lattice velocities and vectors', '  1'])
                for velo in self.lattice_velocities:
                    lines.append(' '.join((f' {val: .7E}' for val in velo)))
            except Exception:
                warnings.warn('Lattice velocities are missing or corrupted.', BadPoscarWarning)
        if self.velocities:
            try:
                lines.append('')
                for velo in self.velocities:
                    lines.append(' '.join((format_str.format(val) for val in velo)))
            except Exception:
                warnings.warn('Velocities are missing or corrupted.', BadPoscarWarning)
        if self.predictor_corrector:
            lines.append('')
            if self.predictor_corrector_preamble:
                lines.append(self.predictor_corrector_preamble)
                pred = np.array(self.predictor_corrector)
                for col in range(3):
                    for z in pred[:, col]:
                        lines.append(' '.join((format_str.format(i) for i in z)))
            else:
                warnings.warn('Preamble information missing or corrupt. Writing Poscar with no predictor corrector data.', BadPoscarWarning)
        return '\n'.join(lines) + '\n'

    def __repr__(self):
        return self.get_str()

    def __str__(self):
        """String representation of Poscar file."""
        return self.get_str()

    def write_file(self, filename: PathLike, **kwargs):
        """
        Writes POSCAR to a file. The supported kwargs are the same as those for
        the Poscar.get_str method and are passed through directly.
        """
        with zopen(filename, mode='wt') as file:
            file.write(self.get_str(**kwargs))

    def as_dict(self) -> dict:
        """MSONable dict."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'structure': self.structure.as_dict(), 'true_names': self.true_names, 'selective_dynamics': np.array(self.selective_dynamics).tolist(), 'velocities': self.velocities, 'predictor_corrector': self.predictor_corrector, 'comment': self.comment}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            Poscar
        """
        return Poscar(Structure.from_dict(dct['structure']), comment=dct['comment'], selective_dynamics=dct['selective_dynamics'], true_names=dct['true_names'], velocities=dct.get('velocities'), predictor_corrector=dct.get('predictor_corrector'))

    def set_temperature(self, temperature: float):
        """
        Initializes the velocities based on Maxwell-Boltzmann distribution.
        Removes linear, but not angular drift (same as VASP).

        Scales the energies to the exact temperature (microcanonical ensemble)
        Velocities are given in A/fs. This is the vasp default when
        direct/cartesian is not specified (even when positions are given in
        direct coordinates)

        Overwrites imported velocities, if any.

        Args:
            temperature (float): Temperature in Kelvin.
        """
        velocities = np.random.randn(len(self.structure), 3)
        atomic_masses = np.array([site.specie.atomic_mass.to('kg') for site in self.structure])
        dof = 3 * len(self.structure) - 3
        velocities -= np.average(atomic_masses[:, np.newaxis] * velocities, axis=0) / np.average(atomic_masses)
        velocities /= atomic_masses[:, np.newaxis] ** (1 / 2)
        energy = np.sum(1 / 2 * atomic_masses * np.sum(velocities ** 2, axis=1))
        scale = (temperature * dof / (2 * energy / const.k)) ** (1 / 2)
        velocities *= scale * 1e-05
        self.temperature = temperature
        self.structure.site_properties.pop('selective_dynamics', None)
        self.structure.site_properties.pop('predictor_corrector', None)
        self.structure.add_site_property('velocities', velocities.tolist())